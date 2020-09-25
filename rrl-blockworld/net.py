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
    samples = [torch.multinomial(x, 1) for x in probs_split]
    
    return torch.cat(samples)

def segmented_scatter_(dest, indices, start_indices, values):
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest

def segmented_gather(src, indices, start_indices):
    real_indices = start_indices + indices
    return src[real_indices]

EMB_SIZE = 32
class Net(Module):
    def __init__(self):
        super().__init__()

        self.embed_node = Sequential( Linear(1, EMB_SIZE), LeakyReLU() )

        self.mp_main = MultiMessagePassing(steps=config.mp_iterations)

        self.a_2 = MultiMessagePassing(steps=2)

        self.sel_enc = Sequential( Linear(EMB_SIZE + 1, EMB_SIZE), LeakyReLU() )

        self.a_1_sel = Linear(EMB_SIZE, 1) # node features -> node probability
        self.a_2_sel = Linear(EMB_SIZE, 1) # node features -> node probability
        self.value_function = Linear(EMB_SIZE, 1) # global features -> state value

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

    def forward(self, s_batch, only_v=False, complete=False):
        node_feats, edge_attr, edge_index, free_boxes = zip(*s_batch)

        node_feats = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in node_feats]
        edge_attr  = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in edge_attr]
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

        def make_mask(lst):
            mask = torch.tensor( np.concatenate(lst), device=self.device )
            lst_lens = torch.tensor([len(x) for x in lst], device=self.device)
            mask_starts = data_starts.repeat_interleave(lst_lens)

            return mask, mask_starts

        # decode first action
        # x_a1, _ = self.a_1(x, xg, batch.edge_attr, batch.edge_index, batch_ind, batch.num_graphs)
        x_a1 = self.a_1_sel(x).flatten()
        mask_a1, mask_starts_a1 = make_mask(free_boxes)        # only the free boxes can be selected as a1
        p_a1 = masked_segmented_softmax(x_a1, mask_a1, mask_starts_a1, batch_ind)   

        # sample & encode first action
        a1 = segmented_sample(p_a1, data_splits)
        selected_ind = torch.zeros(len(batch.x), 1, device=self.device)
        segmented_scatter_(selected_ind, a1, data_starts, 1.0)

        # decode second action
        x = torch.cat((x, selected_ind), dim=1)
        x = self.sel_enc(x) # 33 -> 32

        x_a2, _ = self.a_2(x, xg, batch.edge_attr, batch.edge_index, batch_ind, batch.num_graphs)
        x_a2 = self.a_2_sel(x_a2).flatten()

        boxes_a2 = [list(filter(lambda z: z != a1[i], [0] + boxs)) for i, boxs in enumerate(free_boxes)]
        mask_a2, mask_starts_a2 = make_mask(boxes_a2)          # free a2; a2 != a1

        p_a2 = masked_segmented_softmax(x_a2, mask_a2, mask_starts_a2, batch_ind)  

        # sample second action
        a2 = segmented_sample(p_a2, data_splits)

        # compute the selected action probability
        a1_p = segmented_gather(p_a1, a1, data_starts)
        a2_p = segmented_gather(p_a2, a2, data_starts)
        tot_prob = a1_p * a2_p

        # convert the actions to tuples
        a1 = a1.cpu().numpy()
        a2 = a2.cpu().numpy()
        a = list(zip(a1, a2))

        return a, value, tot_prob

    def update(self, r, v, pi, s_, n_stacks, done, target_net=None):
        done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).view(-1, 1)

        if target_net is None:
            target_net = self

        v_ = target_net(s_, only_v=True) * (1. - done)
        num_actions = torch.tensor(n_stacks, dtype=torch.float32, device=self.device) ** 2
        # num_actions = torch.tensor([x[0].shape[0] ** 2 for x in s_], dtype=torch.float32, device=self.device) # (node * node) actions

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
    def __init__(self, steps, node_in_size=EMB_SIZE, node_out_size=EMB_SIZE, edge_size=2, global_size=EMB_SIZE, agg_size=EMB_SIZE):
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
