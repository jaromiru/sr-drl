import torch, numpy as np
import torch_geometric, torch_scatter

from torch.nn import *
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.data import Data, Batch

from rl import a2c
from config import config

def segmented_sample(probs, splits):
    probs_split = torch.split(probs, splits)
    samples = [torch.multinomial(x, 1) for x in probs_split]
    
    return torch.cat(samples)

class Net(Module):
    def __init__(self):
        super().__init__()

        self.embed_node = Sequential( Linear(5, config.emb_size), LeakyReLU() )
        # self.embed_edge = Sequential( Linear(4, config.emb_size), LeakyReLU() )

        self.message_passing = MultiMessagePassing(steps=config.mp_iterations)

        self.node_select = Linear(config.emb_size, 5) # node features -> node probability for all 5 actions
        self.action_select = Linear(config.emb_size, 5)  # global features -> 5 actions
        self.value_function = Linear(config.emb_size, 1) # global features -> state value

        self.opt = torch.optim.AdamW(self.parameters(), lr=config.opt_lr, weight_decay=config.opt_l2)

        self.device = torch.device(config.device)
        self.to(self.device)

        # auxiliary variables
        # self.one_hot = torch.eye(5).to(self.device)

        self.lr = config.opt_lr
        self.alpha_h = config.alpha_h

    def save(self, file='model.pt'):
        torch.save(self.state_dict(), file)

    def load(self, file='model.pt'):
        self.load_state_dict(torch.load(file, map_location=self.device))

    # def copy_weights(self, other):
    #     params_other = list(other.parameters())

    #     for i in range( len(params_other) ):
    #         val_new   = params_other[i].data
    #         params_self[i].data.copy_(val_new)

    def copy_weights(self, other, rho):
        params_other = list(other.parameters())
        params_self  = list(self.parameters())

        for i in range( len(params_other) ):
            val_self  = params_self[i].data
            val_other = params_other[i].data
            val_new   = rho * val_other + (1-rho) * val_self

            params_self[i].data.copy_(val_new)

    def forward(self, s_batch, only_v=False, complete=False):
        # convert to tensors
        node_feats, edge_attr, edge_index, step_idx, used_indices = zip(*s_batch)

        # the tensors have different length
        node_feats = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in node_feats]
        edge_attr  = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in edge_attr]
        edge_index = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in edge_index]
        # step_idx = torch.tensor(step_idx, dtype=torch.float32, device=self.device)
        step_idx = None

        # create batch
        data = [Data(x=node_feats[i], edge_attr=edge_attr[i], edge_index=edge_index[i]) for i in range( len(s_batch) )]
        data_lens = [x.num_nodes for x in data]
        batch = Batch.from_data_list(data)
        batch_ind = batch.batch.to(self.device) # graph indices in the batch

        # embed features
        x, edge_attr, edge_index = batch.x, batch.edge_attr, batch.edge_index
        x = self.embed_node(x)
        # edge_attr = self.embed_edge(edge_attr)

        # push through graph
        x, xg = self.message_passing(x, step_idx, edge_attr, edge_index, batch_ind, batch.num_graphs)

        # compute value function
        value = self.value_function(xg)

        if only_v:
            return value

        def sample_action(xg):
            out_action = self.action_select(xg)
            action_softmax = torch.distributions.Categorical( torch.softmax(out_action, dim=1) )
            action_selected = action_softmax.sample()

            return action_softmax, action_selected

        def sample_node(x, a):
            a_expanded = a[batch_ind].view(-1, 1)               # a single action is performed for each graph 
            out_node = self.node_select(x)                      # node_select outputs actiovations for each action,
            node_activation = out_node.gather(1, a_expanded)    # hence here we select only the performed action

            node_softmax = torch_geometric.utils.softmax(node_activation.flatten(), batch_ind)
            node_selected = segmented_sample(node_softmax, data_lens)

            # since all the graphs are of the same size, we can simplify things     
            # node_activation = node_activation.view(batch.num_graphs, data[0].num_nodes)
            # node_softmax = torch.distributions.Categorical( torch.softmax(node_activation, dim=1) )
            # node_selected = node_softmax.sample()

            return node_softmax, node_selected

        # return complete probs for debug
        if complete:
            action_softmax, _ = sample_action(xg)

            out_node = self.node_select(x)
            # if complete, there is only one sample
            node_activations = out_node.reshape(batch.num_graphs, data[0].num_nodes, 5)
            node_softmaxes = node_activations.softmax(dim=1)

            return action_softmax, node_softmaxes, value

        # select an action & node
        action_softmax, action_selected = sample_action(xg)
        node_softmax, node_selected = sample_node(x, action_selected)

        # compute the selected action probability
        a_prob = action_softmax.probs.gather(1, action_selected.view(-1, 1))

        # get proper node probability indexes
        data_starts = np.concatenate( ([0], data_lens[:-1]) )
        data_starts = torch.tensor(data_starts, device=self.device, dtype=torch.int64)
        n_index = torch.cumsum(data_starts, 0) + node_selected
        n_prob = node_softmax[n_index].view(-1, 1)

        tot_prob = a_prob * n_prob

        return action_selected, node_selected, value, tot_prob

    def update(self, r, v, pi, s_, done, target_net=None):
        done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).view(-1, 1)

        if target_net is None:
            target_net = self

        v_ = target_net(s_, only_v=True) * (1. - done)

        num_actions = torch.tensor([x[0].shape[0] * 5 for x in s_], dtype=torch.float32, device=self.device).reshape(-1, 1) # 5 actions per node

        loss, loss_pi, loss_v, loss_h, entropy = a2c(r, v, v_, pi, config.gamma, config.alpha_v, self.alpha_h, config.q_range, num_actions)

        self.opt.zero_grad()
        loss.backward()

        # clip the gradient norm
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), config.opt_max_norm)

        self.opt.step()

        # for logging
        return loss, loss_pi, loss_v, loss_h, entropy, norm

    def set_lr(self, lr):
        self.lr = lr

        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def set_alpha_h(self, alpha_h):
        self.alpha_h = alpha_h

# ----------------------------------------------------------------------------------------
class MultiMessagePassing(Module):
    def __init__(self, steps):
        super().__init__()

        self.gnns = ModuleList( [GraphNet() for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode() for i in range(steps)] )

        self.steps = steps

    def forward(self, x, step_idx, edge_attr, edge_index, batch_ind, num_graphs):
        x_global = torch.zeros(num_graphs, config.emb_size, device=config.device)  # this can encode context
        # x_global[:, 0] = step_idx   # include step_idx into context

        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_global, batch_ind)
            x_global = self.pools[i](x_global, x, batch_ind)

        return x, x_global

# ----------------------------------------------------------------------------------------
class GlobalNode(Module):       
    def __init__(self):
        super().__init__()

        att_mask = Linear(config.emb_size, 1)
        att_feat = Sequential( Linear(config.emb_size, config.emb_size), LeakyReLU() )

        self.glob = GlobalAttention(att_mask, att_feat)
        self.tranform = Sequential( Linear(config.emb_size + config.emb_size, config.emb_size), LeakyReLU() )

    def forward(self, xg_old, x, batch):
        xg = self.glob(x, batch)

        xg = torch.cat([xg, xg_old], dim=1)
        xg = self.tranform(xg) + xg_old # skip connection

        return xg

# ----------------------------------------------------------------------------------------
class GraphNet(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

        self.f_mess = Sequential( Linear(config.emb_size + 4, config.emb_size), LeakyReLU() )
        self.f_agg  = Sequential( Linear(config.emb_size + config.emb_size + config.emb_size, config.emb_size), LeakyReLU() )

    def forward(self, x, edge_attr, edge_index, xg, batch):
        xg = xg[batch]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, xg=xg)

    def message(self, x_j, edge_attr):
        z = torch.cat([x_j, edge_attr], dim=1)
        z = self.f_mess(z)

        return z 

    def update(self, aggr_out, x, xg):
        z = torch.cat([x, xg, aggr_out], dim=1)
        z = self.f_agg(z) + x # skip connection

        return z
