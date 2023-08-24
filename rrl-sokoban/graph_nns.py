import torch, torch_geometric
from torch.nn import *
from torch_geometric.nn import MessagePassing, GlobalAttention, TransformerConv, GATv2Conv, LayerNorm, global_mean_pool, global_max_pool

from config import config

# ----------------------------------------------------------------------------------------
class MultiMessagePassing(Module):
    def __init__(self, steps):
        super().__init__()

        self.gnns = ModuleList( [GraphNet() for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode() for i in range(steps)] )

        self.steps = steps

    def forward(self, x, edge_attr, edge_index, batch_ind, num_graphs):
        x_global = torch.zeros(num_graphs, config.emb_size, device=config.device)  # this can encode context

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

    def forward(self, xg_old, x, batch_ind):
        xg = self.glob(x, batch_ind)

        xg = torch.cat([xg, xg_old], dim=1)
        xg = self.tranform(xg) + xg_old # skip connection

        return xg

# ----------------------------------------------------------------------------------------
class GraphNet(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

        self.f_mess = Sequential( Linear(config.emb_size + 4, config.emb_size), LeakyReLU() )
        self.f_agg  = Sequential( Linear(config.emb_size + config.emb_size + config.emb_size, config.emb_size), LeakyReLU() )

    def forward(self, x, edge_attr, edge_index, xg, batch_ind):
        xg = xg[batch_ind]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, xg=xg)

    def message(self, x_j, edge_attr):
        z = torch.cat([x_j, edge_attr], dim=1)
        z = self.f_mess(z)

        return z 

    def update(self, aggr_out, x, xg):
        z = torch.cat([x, xg, aggr_out], dim=1)
        z = self.f_agg(z) + x # skip connection

        return z
