import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MATRIXDECODER(nn.Module):
    def __init__(self, input_size, max_atom_number, len_atom_type, len_bond_type, matrix_size, 
                 num_layers, dropout_ratio, teacher_forcing, device='cpu'):
        super(MATRIXDECODER, self).__init__()
        hidden_size = 256
        # length column
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_atom_number = nn.Linear(hidden_size, max_atom_number)
        # annotation matrix
        self.linear_3 = nn.Linear(input_size+max_atom_number, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, max_atom_number*len_atom_type)
        self.linear_atom_type = nn.Linear(len_atom_type, len_atom_type)
        # inital adjacency matrix
        self.linear_node_embedding = nn.Linear(len_atom_type, 256)
        self.linear_pre_adjacency_matrix = nn.Linear(256, 1)
        # adjacency matrix
        self.linear_7 = nn.Linear(max_atom_number*max_atom_number, max_atom_number*max_atom_number*len_bond_type)
        self.linear_8 = nn.Linear(max_atom_number*len_bond_type, max_atom_number*len_bond_type)
        self.linear_bond_type = nn.Linear(len_bond_type, len_bond_type)
        
        self.max_atom_number = max_atom_number
        self.len_atom_type = len_atom_type
        self.len_bond_type = len_bond_type
        self.matrix_size = matrix_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.teacher_forcing = teacher_forcing
        self.device = device

    def forward(self, z, teacher_tensor=None, guide_tensor=None):

        if teacher_tensor is not None:
            teacher_matrix = teacher_tensor.reshape(-1, self.max_atom_number, self.matrix_size//self.max_atom_number) # [b, N, M//N]

        # max_atom_number matrix # [b, N]
        max_atom_number_output = F.relu(self.linear_1(z)) # [b, 256]
        # max_atom_number_output = F.dropout(max_atom_number_output, self.dropout_ratio, training=self.training) # [b, 256]
        max_atom_number_output = F.relu(self.linear_2(max_atom_number_output)) # [b, 256]
        max_atom_number_output = self.linear_atom_number(max_atom_number_output) # [b, N]

        if self.training and self.teacher_forcing and random.random() < 0.25:
            max_atom_number_temp = teacher_matrix[:, :, 0].to(self.device) # [b, N]
            max_atom_number_temp = max_atom_number_temp.reshape(-1, self.max_atom_number) # [b, N]
        else:
            max_atom_number_temp = F.softmax(max_atom_number_output, dim=1) # [b, N]
            max_atom_number_temp = max_atom_number_temp.reshape(-1, self.max_atom_number) # [b, N]

        # annotation matrix # [b, N, A]
        annotation_matrix_output = torch.cat((z, max_atom_number_temp), dim=1) # [b, 256+N]
        annotation_matrix_output = F.relu(self.linear_3(annotation_matrix_output)) # [b, 256]
        # annotation_matrix_output = F.dropout(annotation_matrix_output, self.dropout_ratio, training=self.training) # [b, 256]
        annotation_matrix_output = F.relu(self.linear_4(annotation_matrix_output)) # [b, N*A]
            
        annotation_matrix_output = annotation_matrix_output.reshape(-1, self.max_atom_number, self.len_atom_type) # [b, N, A]
        annotation_matrix_output = self.linear_atom_type(annotation_matrix_output) # [b, N, A]

        if self.training and self.teacher_forcing and random.random() < 0.25:
            annotation_matrix_temp = teacher_matrix[:, :, 1:self.len_atom_type+1].to(self.device) # [b, N, A]
            annotation_matrix_temp = annotation_matrix_temp.reshape(-1, self.max_atom_number*self.len_atom_type) # [b, N*A]
        else:
            annotation_matrix_temp = annotation_matrix_output.reshape(-1, self.max_atom_number, self.len_atom_type) # [b, N, A]
            annotation_matrix_temp = F.softmax(annotation_matrix_temp, dim=2).reshape(-1, self.max_atom_number*self.len_atom_type) # [b, N*A]

        # pre_adjacency matrix # [b, N, N, 1]
        node_for_adjacency_output = annotation_matrix_temp.reshape(-1, self.max_atom_number, self.len_atom_type) # [b, N, A]
        node_for_adjacency_output = self.linear_node_embedding(node_for_adjacency_output) # [b, N, 256]
        node_for_adjacency_output = node_for_adjacency_output.unsqueeze(2) * node_for_adjacency_output.unsqueeze(1) # [b, N, N, 256]

        if self.training and self.teacher_forcing and random.random() < 0.25:
            pre_adjacency_matrix_temp = teacher_matrix[:, :, self.len_atom_type+1:].to(self.device) # [b, N, N, B]
            pre_adjacency_matrix_temp = pre_adjacency_matrix_temp.reshape(-1, self.max_atom_number, self.len_bond_type) # [b, N, B]
            pre_adjacency_matrix_temp = torch.argmax(pre_adjacency_matrix_temp, axis=2).type(torch.FloatTensor).to(self.device) # [b, N]
            # pre_adjacency_matrix_temp = torch.where(pre_adjacency_matrix_temp>0, 1, pre_adjacency_matrix_temp).type(torch.FloatTensor).to(self.device)
            pre_adjacency_matrix_temp = pre_adjacency_matrix_temp.reshape(-1, self.max_atom_number, self.max_atom_number) # [b, N, N]
        else:
            pre_adjacency_matrix_temp = self.linear_pre_adjacency_matrix(node_for_adjacency_output) # [b, N, N, 1]
            pre_adjacency_matrix_temp = F.relu(pre_adjacency_matrix_temp) # [b, N, N, 1]
            pre_adjacency_matrix_temp = pre_adjacency_matrix_temp.reshape(-1, self.max_atom_number, self.max_atom_number, 1) # [b, N, N, 1]
            pre_adjacency_matrix_temp = (pre_adjacency_matrix_temp + pre_adjacency_matrix_temp.permute(0, 2, 1, 3)) / 2 # [b, N, N, 1]

        # adjacency matrix # [b, N, N, B]
        adjacency_matrix_output = pre_adjacency_matrix_temp.reshape(-1, self.max_atom_number*self.max_atom_number) # [b, N*N]
        adjacency_matrix_output = F.relu(self.linear_7(adjacency_matrix_output)) # [b, N*N*B]
        # decoder_output = F.dropout(decoder_output, self.dropout_ratio, training=self.training)
        adjacency_matrix_output = adjacency_matrix_output.reshape(-1, self.max_atom_number, self.max_atom_number*self.len_bond_type) # [b, N, N*B]
        adjacency_matrix_output = F.relu(self.linear_8(adjacency_matrix_output)) # [b, N, N*B]

        adjacency_matrix_output = adjacency_matrix_output.reshape(-1, self.max_atom_number, self.max_atom_number, self.len_bond_type) # [b, N, N, B]
        adjacency_matrix_output = self.linear_bond_type(adjacency_matrix_output) # [b, N, N, B]
        adjacency_matrix_output = (adjacency_matrix_output + adjacency_matrix_output.permute(0, 2, 1, 3)) / 2 # [b, N, N, B]
        
        # final matrix # [b, M]
        decoder_output_final = torch.cat((max_atom_number_output.reshape(-1, self.max_atom_number, 1),
                                          annotation_matrix_output.reshape(-1, self.max_atom_number, self.len_atom_type), 
                                          adjacency_matrix_output.reshape(-1, self.max_atom_number, self.max_atom_number*self.len_bond_type)), dim=2) # [b, N, M//N]
        decoder_output_final = decoder_output_final.reshape(-1, self.matrix_size) # [b, M]
    
        return decoder_output_final