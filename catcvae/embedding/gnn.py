import torch
import torch.nn as nn
from torch.nn import Linear, GRUCell
from torch_geometric.nn import GCNConv, GATConv, GINConv, GCNConv, SAGEConv # GINEConv (error if edge_attr is None)
# from catcvae.embedding.gineconv import GINEConv # used modified version from torch_geometric
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F
from catcvae.molgraph import num_node_features, num_edge_features

class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", readout="add", dropout_ratio=0., gnn_type="gat", device='cpu'):
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        super(GNN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.num_layer = num_layer
        self.JK = JK
        self.readout = readout
        self.device = device

        # pre-embedding
        # node
        self.x_linear = nn.Linear(num_node_features, emb_dim)
        # edge
        # self.edge_linear = nn.Linear(num_edge_features, num_edge_features)

        # node embedding
        # gnn architecture
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            # elif gnn_type == "gine":
            #     self.gnns.append(GINEConv(GINE_Sequential(in_channels=emb_dim, hidden_channels=emb_dim, dropout=dropout_ratio), aggr="add", edge_dim=num_edge_features))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(in_channels=emb_dim, out_channels=emb_dim, edge_dim=num_edge_features))
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv(in_channels=emb_dim, out_channels=emb_dim))
        # batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        # graph embedding
        if self.readout == "gru":
            self.mol_conv = GATConv(emb_dim, emb_dim, add_self_loops=False)
            self.mol_bns = nn.BatchNorm1d(emb_dim)
            self.mol_gru = GRUCell(emb_dim, emb_dim)
            # self.mol_lin = Linear(in_lin, out_lin)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        # pre-embedding
        x = self.x_linear(x)
        # edge_attr = self.edge_linear(edge_attr) if edge_attr.shape[0] != 0 else None
        edge_attr = edge_attr if edge_attr.shape[0] != 0 else None

        # node embedding
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr=edge_attr)
            if ((h.size(0) != 1 and self.training) or (not self.training)): h = self.batch_norms[layer](h)
            # remove relu for the last layer
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.dropout_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        
        # graph embedding
        if self.readout == "mean":
            graph_representation = global_mean_pool(node_representation, batch)
        elif self.readout == "add":
            graph_representation = global_add_pool(node_representation, batch)
        elif self.readout == "max":
            graph_representation = global_max_pool(node_representation, batch)
        elif self.readout == "gru":
            # att_mol_stack = list()
            row = torch.arange(batch.size(0), device=batch.device)
            edge_index = torch.stack([row, batch], dim=0)
            out = F.leaky_relu(global_add_pool(node_representation, batch))
            for t in range(self.num_layer):
                h, attention_weights = self.mol_conv((x, out), edge_index, return_attention_weights=True)
                if ((h.size(0) != 1 and self.training) or (not self.training)): h = self.mol_bns(h)
                h = F.elu_(h)
                h = F.dropout(h, p=self.dropout_ratio, training=self.training)
                out = self.mol_gru(h, out)
                out = F.leaky_relu(out)
                # att_mol_index, att_mol_weights = attention_weights
                # att_mol_stack.append(att_mol_weights)
            # Predictor:
            # out = F.dropout(out, p=self.dropout_ratio, training=self.training)
            # graph_representation = self.mol_lin(out)
            graph_representation = out
            # mean of attention weight
            # att_mol_mean = torch.mean(torch.stack(att_mol_stack), dim=0)
            
        return node_representation, graph_representation
    
class GINE_Sequential(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super(GINE_Sequential, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x