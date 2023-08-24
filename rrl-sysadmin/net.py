import torch, numpy as np
import torch_geometric, torch_scatter

from torch.nn import *
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.data import Data, Batch

from rl import a2c
from config import config

def get_start_indices(splits):
    splits = torch.roll(splits, 1)
    splits[0] = 0

    start_indices = torch.cumsum(splits, 0)
    return start_indices

# TODO: update to data_starts
def masked_segmented_softmax(energies, mask, start_indices, batch_ind):
    mask = mask + start_indices
    mask_bool = torch.ones_like(energies, dtype=torch.bool) # inverse mask matrix
    mask_bool[mask] = False

    energies[mask_bool] = -np.inf
    probs = torch_geometric.utils.softmax(energies, batch_ind) # to probs ; per graph

    return probs

def segmented_sample(probs, splits):
    probs_split = torch.split(probs, splits)
    # print(probs_split)

    samples = [torch.multinomial(x, 1) for x in probs_split]
    
    return torch.cat(samples)

def segmented_prod(tnsr, splits):
    x_split = torch.split(tnsr, splits)
    x_prods = [torch.prod(x) for x in x_split]
    x_mul = torch.stack(x_prods)

    return x_mul

def segmented_nonzero(tnsr, splits):
    x_split = torch.split(tnsr, splits)
    x_nonzero = [torch.nonzero(x, as_tuple=False).flatten().cpu().tolist() for x in x_split]

    return x_nonzero

# def segmented_sample_with_mask(energies, splits):
#     energies_split = torch.split(energies, splits)
#     probs_split = [torch.softmax(x) for x in energies_split]
#     samples = [torch.multinomial(x, 1) for x in probs_split]
    
#     return torch.cat(samples)

def segmented_scatter_(dest, indices, start_indices, values):
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest

def segmented_gather(src, indices, start_indices):
    real_indices = start_indices + indices
    return src[real_indices]

EMB_SIZE = 32
class Net(Module):
    def __init__(self, multi=False):
        super().__init__()

        self.embed_node = Sequential( Linear(1, EMB_SIZE), LeakyReLU() )
        # self.embed_edge = Sequential( Linear(4, EMB_SIZE), LeakyReLU() )

        self.mp_main = MultiMessagePassing(steps=config.mp_iterations)

        if multi:
            self.a_0_sel = Sequential( Linear(EMB_SIZE, 1), Sigmoid() ) # Bernoulli trial for each node

        else:
            self.a_0_sel = Linear(EMB_SIZE, 2) # select, no-op
            self.a_1_sel = Linear(EMB_SIZE, 1) # node features -> node probability

        self.value_function = Linear(EMB_SIZE, 1) # global features -> state value

        self.opt = torch.optim.AdamW(self.parameters(), lr=config.opt_lr, weight_decay=config.opt_l2)

        self.device = torch.device(config.device)
        self.to(self.device)

        self.lr = config.opt_lr
        self.alpha_h = config.alpha_h

        self.multi = multi

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
        node_feats, edge_attr, edge_index = zip(*s_batch)

        node_feats = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in node_feats]
        edge_attr  = [torch.empty(0, dtype=torch.float32, device=self.device) for x in edge_attr]
        edge_index = [torch.tensor(x, dtype=torch.int64, device=self.device) for x in edge_index]

        # create batch
        def convert_batch(feats, edge_attr, edge_index):
            data = [Data(x=feats[i], edge_attr=edge_attr[i], edge_index=edge_index[i]) for i in range( len(feats) )]
            batch = Batch.from_data_list(data)
            batch_ind = batch.batch.to(self.device)

            return data, batch, batch_ind

        data, batch, batch_ind = convert_batch(node_feats, edge_attr, edge_index)

        # process state
        x = self.embed_node(batch.x)
        xg = torch.zeros(batch.num_graphs, EMB_SIZE, device=config.device)

        x, xg = self.mp_main(x, xg, batch.edge_attr, batch.edge_index, batch_ind, batch.num_graphs)

        # decode value
        value = self.value_function(xg)

        if only_v:
            return value

        # aux vars
        data_splits = [x.num_nodes for x in data]
        data_splits_tensor = torch.tensor(data_splits, device=self.device)
        data_starts = get_start_indices(data_splits_tensor)

        # ========== MULTI-SELECT ===========
        if self.multi:
            a0_act = self.a_0_sel(x).flatten()
            a0_sel = a0_act.bernoulli().to(torch.uint8)

            # construct selected actions lists
            af_selection = segmented_nonzero(a0_sel, data_splits)

            # get the policy prob
            a0_prob = torch.where(a0_sel, a0_act, 1-a0_act)
            af_probs = segmented_prod(a0_prob, data_splits)

            # print()
            # print(a0_act)
            # print(a0_sel)
            # print(af_selection)
            # print(a0_prob)
            # print(af_probs)
            # exit()

            node_probs = a0_act

        # ========== SINGLE-SELECT ===========
        else:
            # choose a_0
            a0_activation = self.a_0_sel(xg)
            a0_softmax = torch.distributions.Categorical( torch.softmax(a0_activation, dim=1) )
            a0_selection = a0_softmax.sample() # 0 = select; 1 = noop
            a0_selection_tensor = a0_selection.type(torch.uint8).detach()

            # select node
            a1_activation = self.a_1_sel(x)
            a1_softmax = torch_geometric.utils.softmax(a1_activation.flatten(), batch_ind)
            a1_selection = segmented_sample(a1_softmax, data_splits)
            a1_probs = segmented_gather(a1_softmax, a1_selection, data_starts)

            # get final probs
            # af_selection_x = np.where(a0_selection.cpu(), -1, a1_selection.cpu())

            a0_cpu = a0_selection.cpu().numpy()
            a1_cpu = a1_selection.cpu().numpy()

            af_selection = [[] if x else [a1_cpu[i].item()] for i, x in enumerate(a0_cpu)]
            af_probs = torch.where(a0_selection_tensor, a0_softmax.probs[:, 1], a0_softmax.probs[:, 0] * a1_probs)

            node_probs = a1_softmax

        return af_selection, value, af_probs, node_probs # todo, add noop prob

    def update(self, r, v, pi, s_, num_obj, done, target_net=None):
        done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).view(-1, 1)

        if target_net is None:
            target_net = self

        v_ = target_net(s_, only_v=True) * (1. - done)

        # if self.multi:
        #     log_num_actions = torch.tensor(num_obj, dtype=torch.float32, device=self.device) * 0.693147 # log(2)
        #     # log_num_actions[:] = 41. * np.log(2) # fix the number of object to 41

        # else:
        #     log_num_actions = (torch.tensor(num_obj, dtype=torch.float32, device=self.device) + 1).log() # one noop
        #     # log_num_actions[:] = np.log(41) # fix the number of object to 41

        log_num_actions = None # disable entropy scaling
        # print(log_num_actions)

        loss, loss_pi, loss_v, loss_h, entropy = a2c(r, v, v_, pi, config.gamma, config.alpha_v, self.alpha_h, config.q_range, log_num_actions)

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
    def __init__(self, steps, node_in_size=EMB_SIZE, node_out_size=EMB_SIZE, edge_size=0, global_size=EMB_SIZE, agg_size=EMB_SIZE):
        super().__init__()

        # if node_in_size is None:
        #     node_in_size = [EMB_SIZE] * size
            

        self.gnns = ModuleList( [GraphNet(node_in_size, edge_size, global_size, agg_size, node_out_size) for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode(node_out_size, global_size) for i in range(steps)] )

        self.steps = steps

    def forward(self, x, x_global, edge_attr, edge_index, batch_ind, num_graphs):
        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_global, batch_ind)
            x_global = self.pools[i](x_global, x, batch_ind)

        return x, x_global

# ----------------------------------------------------------------------------------------
class GlobalNode(Module):       
    def __init__(self, node_size, global_size):
        super().__init__()

        att_mask = Linear(node_size, 1)
        att_feat = Sequential( Linear(node_size, node_size), LeakyReLU() )

        self.glob = GlobalAttention(att_mask, att_feat)
        self.tranform = Sequential( Linear(global_size * 2, global_size), LeakyReLU() )

    def forward(self, xg_old, x, batch):
        xg = self.glob(x, batch)

        xg = torch.cat([xg, xg_old], dim=1)
        xg = self.tranform(xg) + xg_old # skip connection

        return xg

# ----------------------------------------------------------------------------------------
class GraphNet(MessagePassing):
    def __init__(self, node_in_size, edge_size, global_size, agg_size, node_out_size):
        super().__init__(aggr='max')

        self.f_mess = Sequential( Linear(node_in_size + edge_size, agg_size), LeakyReLU() )
        self.f_agg  = Sequential( Linear(node_in_size + global_size + agg_size, node_out_size), LeakyReLU() )

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
