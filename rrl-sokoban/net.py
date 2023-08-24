# TODOs:
# - preconditions for actions (only appliable if there is/isn't a box)

import torch, numpy as np
import torch_geometric, torch_scatter

from torch.nn import *
from torch_geometric.data import Data, Batch

from rl import a2c
from config import config

from graph_nns import *

def segmented_sample(probs, splits):
    probs_split = torch.split(probs, splits)
    samples = [torch.multinomial(x, 1) for x in probs_split]
    
    return torch.cat(samples)

class Net(Module):
    def __init__(self):
        super().__init__()

        if config.pos_feats:
            node_feat_size = 5
        else:
            node_feat_size = 3

        self.embed_node = Sequential( Linear(node_feat_size, config.emb_size), LeakyReLU() )
        # self.embed_edge = Sequential( Linear(4, config.emb_size), LeakyReLU() )

        self.gnn = MultiMessagePassing(config.mp_iterations)

        # self.gnn = MultiAlternatingTransformer(input_size=config.emb_size, output_size=config.emb_size, edge_size=4, layers=config.mp_iterations, heads=config.att_heads, mean_at_end=True)
        # self.gnn = MultiTransformer(input_size=config.emb_size, output_size=config.emb_size, edge_size=4, layers=config.mp_iterations, heads=config.att_heads, mean_at_end=True)
        # self.pooling = GlobalPooling()

        # self.node_select = Sequential(Linear(config.emb_size, config.emb_size), LeakyReLU(), Linear(config.emb_size, 5)) # node features -> node probability for all 5 actions
        # self.action_select = Sequential(Linear(config.emb_size * 2, config.emb_size), LeakyReLU(), Linear(config.emb_size, 5))  # global features -> 5 actions
        # self.value_function = Sequential(Linear(config.emb_size * 2, config.emb_size), LeakyReLU(), Linear(config.emb_size, 1)) # global features -> state value

        self.node_select = Linear(config.emb_size, 4) # node features -> node probability for all 4 actions (without the move action)
        self.action_select = Linear(config.emb_size, 4)  # global features -> 4 actions 
        self.value_function = Linear(config.emb_size, 1) # global features -> state value

        self.opt = torch.optim.AdamW(self.parameters(), lr=config.opt_lr, weight_decay=config.opt_l2)

        self.device = torch.device(config.device)
        self.to(self.device)

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

    @staticmethod
    def prepare_batch(s_batch):
        node_feats, edge_attr, edge_index, step_idx, used_indices = zip(*s_batch)

        node_feats = [torch.tensor(x, dtype=torch.float32, device=config.device) for x in node_feats]
        edge_attr  = [torch.tensor(x, dtype=torch.float32, device=config.device) for x in edge_attr]
        edge_index = [torch.tensor(x, dtype=torch.int64, device=config.device) for x in edge_index]

        # create batch
        data = [Data(x=node_feats[i], edge_attr=edge_attr[i], edge_index=edge_index[i]) for i in range( len(s_batch) )]
        data_lens = [x.num_nodes for x in data]
        batch = Batch.from_data_list(data)
        batch_ind = batch.batch.to(config.device) # graph indices in the batch

        return data, data_lens, batch, batch_ind

    def forward(self, s_batch, only_v=False, complete=False):
        data, data_lens, batch, batch_ind = self.prepare_batch(s_batch)

        # process state
        x = self.embed_node(batch.x)
        # x = self.gnn(x, batch.edge_index, batch.edge_attr)
        x, x_pooled = self.gnn(x, batch.edge_attr, batch.edge_index, batch_ind, batch.num_graphs)
        # x_pooled = self.pooling(x, batch_ind)

        # decode value
        value = self.value_function(x_pooled)

        if only_v:
            return value

        def sample_action(x_pooled):
            out_action = self.action_select(x_pooled)
            action_softmax = torch.distributions.Categorical( torch.softmax(out_action, dim=1) )
            action_selected = action_softmax.sample()

            return action_softmax, action_selected

        def sample_node(x, a):
            a_expanded = a[batch_ind].view(-1, 1)               # a single action is performed for each graph 
            out_node = self.node_select(x)                      # node_select outputs actiovations for each action,
            node_activation = out_node.gather(1, a_expanded)    # hence here we select only the performed action

            # enable selection of boxes only
            if config.precond:
                nobox_mask = batch.x[:, 1] == 0.
                node_activation[nobox_mask] = -np.inf

            node_softmax = torch_geometric.utils.softmax(node_activation.flatten(), batch_ind)
            node_selected = segmented_sample(node_softmax, data_lens)
            
            # since all the graphs are of the same size, we can simplify things     
            # node_activation = node_activation.view(batch.num_graphs, data[0].num_nodes)
            # node_softmax = torch.distributions.Categorical( torch.softmax(node_activation, dim=1) )
            # node_selected = node_softmax.sample()

            return node_softmax, node_selected

        # return complete probs for debug; assuming only one graph
        if complete:
            action_softmax, _ = sample_action(x_pooled)

            out_node = self.node_select(x)     

            # enable selection of boxes only
            if config.precond:
                nobox_mask = batch.x[:, 1] == 0.
                out_node[nobox_mask] = -np.inf       

            node_activations = out_node.reshape(batch.num_graphs, data[0].num_nodes, 4)
            node_softmaxes = node_activations.softmax(dim=1)
            node_softmaxes = torch.nan_to_num(node_softmaxes)

            return action_softmax, node_softmaxes, value

        # select an action & node
        action_softmax, action_selected = sample_action(x_pooled)
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

        # num_actions = torch.tensor([x[0].shape[0] * 5 for x in s_], dtype=torch.float32, device=self.device).reshape(-1, 1) # 5 actions per node
        num_actions = None # disable entropy scaling

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

