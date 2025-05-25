import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MATRIXENCODER(nn.Module):
    def __init__(self, input_size, max_atom_number, len_atom_type, len_bond_type, 
                 matrix_size, num_layers, dropout_ratio, device='cpu'):
        super(MATRIXENCODER, self).__init__()
        # length column
        self.linear_atom_number = nn.Linear(max_atom_number, input_size)
        self.linear_2 = nn.Linear(input_size, input_size)
        self.linear_1 = nn.Linear(input_size, input_size)
        # annotation matrix
        self.linear_atom_type = nn.Linear(len_atom_type, len_atom_type)
        self.linear_4 = nn.Linear(max_atom_number*len_atom_type, input_size)
        self.linear_3 = nn.Linear(input_size, input_size)
        # adjacency matrix
        self.linear_bond_type = nn.Linear(len_bond_type, len_bond_type)
        self.linear_8 = nn.Linear(max_atom_number*len_bond_type, max_atom_number*len_bond_type)
        self.linear_7 = nn.Linear(max_atom_number*max_atom_number*len_bond_type, input_size)

        self.matrix_size = matrix_size
        self.max_atom_number = max_atom_number
        self.len_atom_type = len_atom_type
        self.len_bond_type = len_bond_type
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.device = device

    def forward(self, x):

        x = x.reshape(-1, self.max_atom_number, 1+self.len_atom_type+(self.max_atom_number*self.len_bond_type))

        # max_atom_number matrix # [b, N]
        encoded_max_atom_number = x[:, :, 0].to(self.device) # [b, 1, N]
        encoded_max_atom_number = encoded_max_atom_number.reshape(-1, self.max_atom_number) # [b, N]
        encoded_max_atom_number = F.relu(self.linear_atom_number(encoded_max_atom_number)) # [b, 256]
        encoded_max_atom_number = F.dropout(encoded_max_atom_number, self.dropout_ratio, training=self.training) # [b, 256]
        encoded_max_atom_number = F.relu(self.linear_2(encoded_max_atom_number)) # [b, 256]
        encoded_max_atom_number = F.dropout(encoded_max_atom_number, self.dropout_ratio, training=self.training) # [b, 256]
        encoded_max_atom_number = self.linear_1(encoded_max_atom_number) # [b, 256]

        # annotation matrix # [b, N, A]
        encoded_annotation_matrix = x[:, :, 1:self.len_atom_type+1] # [b, N*A]
        encoded_annotation_matrix = encoded_annotation_matrix.reshape(-1, self.len_atom_type) # [b, N, A]
        encoded_annotation_matrix = F.relu(self.linear_atom_type(encoded_annotation_matrix)) # [b, N, A]
        encoded_annotation_matrix = F.dropout(encoded_annotation_matrix, self.dropout_ratio, training=self.training) # [b, N, A]
        encoded_annotation_matrix = encoded_annotation_matrix.reshape(-1, self.len_atom_type*self.max_atom_number) # [b, N*A]
        encoded_annotation_matrix = F.relu(self.linear_4(encoded_annotation_matrix)) # [b, 256]
        encoded_annotation_matrix = F.dropout(encoded_annotation_matrix, self.dropout_ratio, training=self.training) # [b, 256]
        encoded_annotation_matrix = self.linear_3(encoded_annotation_matrix) # [b, 256]

        # adjacency matrix # [b, N, N]
        encoded_adjacency_matrix = x[:, :, self.len_atom_type+1:] # [b, N*N*B]
        encoded_adjacency_matrix = encoded_adjacency_matrix.reshape(-1, self.len_bond_type) # [b, N*N, B]
        encoded_adjacency_matrix = F.relu(self.linear_bond_type(encoded_adjacency_matrix)) # [b, N*N, B]
        encoded_adjacency_matrix = F.dropout(encoded_adjacency_matrix, self.dropout_ratio, training=self.training) # [b, N*N, B]
        encoded_adjacency_matrix = encoded_adjacency_matrix.reshape(-1, self.len_bond_type*self.max_atom_number) # [b, N, N*B]
        encoded_adjacency_matrix = F.relu(self.linear_8(encoded_adjacency_matrix)) # [b, N, N*B]
        encoded_adjacency_matrix = F.dropout(encoded_adjacency_matrix, self.dropout_ratio, training=self.training) # [b, N, N*B]
        encoded_adjacency_matrix = encoded_adjacency_matrix.reshape(-1, self.max_atom_number*self.max_atom_number*self.len_bond_type) # [b, N*N*B]
        encoded_adjacency_matrix = self.linear_7(encoded_adjacency_matrix) # [b, 256]
        
        # sum along the last dimension
        decoder_output = encoded_max_atom_number + encoded_annotation_matrix + encoded_adjacency_matrix # [b, 256]

        return decoder_output