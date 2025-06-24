# code adapted from 
# https://github.com/woshixuhao/Retention-Time-Prediction-for-Chromatographic-Enantioseparation

import warnings
import torch
from torch_geometric.nn import MessagePassing
from utils import *
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set
    )

import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


class AtomEncoder(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        full_atom_feature_dims
        ):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.full_atom_feature_dims = full_atom_feature_dims

        for i, dim in enumerate(self.full_atom_feature_dims):
            # Embedding
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(
        self,
        emb_dim,
        full_bond_feature_dims
        ):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.full_bond_feature_dims = full_bond_feature_dims

        for i, dim in enumerate(self.full_bond_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class RBF(torch.nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = centers.reshape([1, -1])
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.reshape([-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class BondFloatRBF(torch.nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (nn.Parameter(torch.arange(0, 2, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
                # (centers, gamma)
                'prop': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'diameter': (nn.Parameter(torch.arange(3, 12, 0.3)), nn.Parameter(torch.Tensor([1.0]))),
                ##=========Only for pure GNN===============
                'column_TPSA': (nn.Parameter(torch.arange(0, 1, 0.05).to(torch.float32)), nn.Parameter(torch.Tensor([1.0]))),
                'column_RASA': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'column_RPSA': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'column_MDEC': (nn.Parameter(torch.arange(0, 10, 0.5)), nn.Parameter(torch.Tensor([2.0]))),
                'column_MATS': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers.to(device), gamma.to(device))
            self.rbf_list.append(rbf)
            linear = torch.nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[:, i].reshape(-1, 1)
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondAngleFloatRBF(torch.nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (nn.Parameter(torch.arange(0, torch.pi, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_angle_float_names:
            if name == 'bond_angle':
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers.to(device), gamma.to(device))
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim)
                self.linear_list.append(linear)
            else:
                linear = nn.Linear(len(self.bond_angle_float_names) - 1, embed_dim)
                self.linear_list.append(linear)
                break

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            if name == 'bond_angle':
                x = bond_angle_float_features[:, i].reshape(-1, 1)
                rbf_x = self.rbf_list[i](x)
                out_embed += self.linear_list[i](rbf_x)
            else:
                x = bond_angle_float_features[:, 1:]
                out_embed += self.linear_list[i](x)
                break
        return out_embed


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                                 nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = edge_attr
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# GNN to generate node embedding
class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        bond_float_names,
        bond_angle_float_names,
        bond_id_names,
        full_atom_feature_dims,
        full_bond_feature_dims,
        num_layers,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        Use_geometry_enhanced=True
        ):
        """GIN Node Embedding Module
        
        """

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(
            emb_dim,
            full_atom_feature_dims
            )
        self.bond_encoder = BondEncoder(
            emb_dim,
            full_bond_feature_dims
            )
        self.bond_float_encoder = BondFloatRBF(
            bond_float_names,
            emb_dim
            )
        self.bond_angle_encoder = BondAngleFloatRBF(
            bond_angle_float_names,
            emb_dim
            )

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.convs_bond_angle = torch.nn.ModuleList()
        self.convs_bond_float = torch.nn.ModuleList()
        self.convs_bond_embeding = torch.nn.ModuleList()
        self.convs_angle_float = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_ba = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.convs_bond_angle.append(GINConv(emb_dim))
            self.convs_bond_embeding.append(
                BondEncoder(
                    emb_dim,
                    full_bond_feature_dims
                    )
                    )
            self.convs_bond_float.append(
                BondFloatRBF(bond_float_names, emb_dim)
                )
            self.convs_angle_float.append(
                BondAngleFloatRBF(bond_angle_float_names, emb_dim)
                )
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_ba.append(torch.nn.BatchNorm1d(emb_dim))
        self.Use_geometry_enhanced = Use_geometry_enhanced
        self.Use_geometry_enhanced = Use_geometry_enhanced
        self.bond_id_names = bond_id_names
    
    def forward(self, batched_atom_bond, batched_bond_angle):
        x, edge_index, edge_attr = (
            batched_atom_bond.x,
            batched_atom_bond.edge_index,
            batched_atom_bond.edge_attr
            )
        edge_index_ba, edge_attr_ba = (
            batched_bond_angle.edge_index,
            batched_bond_angle.edge_attr
            )
        
        # computing input node embedding
        h_list = [self.atom_encoder(x)]  # convert categorical atomic attributes into atom embeddings

        if self.Use_geometry_enhanced:
            h_list_ba = [self.bond_float_encoder(
                edge_attr[:,len(self.bond_id_names):edge_attr.shape[1]+1].to(torch.float32)
                ) + self.bond_encoder(edge_attr[:, 0:len(self.bond_id_names)].to(torch.int64))]
            for layer in range(self.num_layers):
                h = self.convs[layer](h_list[layer], edge_index, h_list_ba[layer])
                cur_h_ba = self.convs_bond_embeding[layer](
                    edge_attr[:, 0:len(self.bond_id_names)].to(torch.int64)) + self.convs_bond_float[layer](edge_attr[:,len(self.bond_id_names):edge_attr.shape[1]+1].to(torch.float32)
                    )
                cur_angle_hidden = self.convs_angle_float[layer](edge_attr_ba)
                h_ba = self.convs_bond_angle[layer](cur_h_ba, edge_index_ba, cur_angle_hidden)

                if layer == self.num_layers - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                    h_ba = F.dropout(h_ba, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(
                        F.relu(h),
                        self.drop_ratio,
                        training=self.training
                        )
                    h_ba = F.dropout(F.relu(h_ba), self.drop_ratio, training=self.training)
                if self.residual:
                    h += h_list[layer]
                    h_ba += h_list_ba[layer]
                h_list.append(h)
                h_list_ba.append(h_ba)


            # Different implementations of Jk-concat
            if self.JK == "last":
                node_representation = h_list[-1]
                edge_representation = h_list_ba[-1]
            elif self.JK == "sum":
                node_representation = 0
                edge_representation = 0
                for layer in range(self.num_layers + 1):
                    node_representation += h_list[layer]
                    edge_representation += h_list_ba[layer]

            return node_representation, edge_representation
        
        if not self.Use_geometry_enhanced:
            for layer in range(self.num_layers):
                h = self.convs[layer](h_list[layer], edge_index,
                                      self.convs_bond_embeding[layer](edge_attr[:, 0:len(self.bond_id_names)].to(torch.int64)) +
                                      self.convs_bond_float[layer](
                                          edge_attr[:, len(self.bond_id_names):edge_attr.shape[1] + 1].to(torch.float32)))
                h = self.batch_norms[layer](h)
                if layer == self.num_layers - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

                if self.residual:
                    h += h_list[layer]

                h_list.append(h)

            # Different implementations of Jk-concat
            if self.JK == "last":
                node_representation = h_list[-1]
            elif self.JK == "sum":
                node_representation = 0
                for layer in range(self.num_layers + 1):
                    node_representation += h_list[layer]

            return node_representation


class GINGraphPooling(nn.Module):

    def __init__(
        self,
        bond_float_names,
        bond_angle_float_names,
        bond_id_names,
        full_atom_feature_dims,
        full_bond_feature_dims,
        num_tasks=1,
        num_layers=5,
        emb_dim=300,
        residual=False,
        drop_ratio=0,
        JK="last",
        graph_pooling="attention",
        descriptor_dim=1781,
        Use_geometry_enhanced=True
        ):
        """GIN Graph Pooling Module

        GINNodeEmbedding (graph representation)

        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表示的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. "sum", "mean", "max", "attention set2set". Defaults to "sum".

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.descriptor_dim = descriptor_dim
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GINNodeEmbedding(
            bond_float_names,
            bond_angle_float_names,
            bond_id_names,
            full_atom_feature_dims,
            full_bond_feature_dims,
            num_layers,
            emb_dim,
            drop_ratio,
            JK,
            residual,
            Use_geometry_enhanced
            )
        self.Use_geometry_enhanced = Use_geometry_enhanced

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool

        elif graph_pooling == "mean":
            self.pool = global_mean_pool

        elif graph_pooling == "max":
            self.pool = global_max_pool

        elif graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, 1)
                    )
                    )

        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

        self.NN_descriptor = nn.Sequential(nn.Linear(self.descriptor_dim, self.emb_dim),
                                           nn.Sigmoid(),
                                           nn.Linear(self.emb_dim, self.emb_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, batched_atom_bond, batched_bond_angle):
        if self.Use_geometry_enhanced:
            h_node, h_node_ba = self.gnn_node(
                batched_atom_bond,
                batched_bond_angle
                )
        else:
            h_node = self.gnn_node(
                batched_atom_bond,
                batched_bond_angle
                )
        h_graph = self.pool(h_node, batched_atom_bond.batch)
        output = self.graph_pred_linear(h_graph)
        if self.training:
            return (output, h_graph)
        else:
            # At inference time, relu is applied to output to ensure positivity
            return (torch.clamp(output, min=0, max=1e8), h_graph)


class ANN(nn.Module):
    '''
    Construct artificial neural network
    '''
    def __init__(self, in_neuron, hidden_neuron, out_neuron):
        super(ANN, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.sigmoid(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.output_layer(x)
        return x
