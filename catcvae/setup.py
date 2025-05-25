import argparse
import os
import torch
import numpy as np
import random as random
from catcvae.molgraph import atom_encoder_m, bond_encoder_m, atom_decoder_m, bond_decoder_m, max_atom_number, matrix_size
from catcvae.condition import getConditionDim, getOneHotCondition
from dataset import _dataset

# parser
class ModelArgumentParser:
    def __init__(self):
        # initialize arguments argparse
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--file', type=str, default='ord')
        self.parser.add_argument('--seed', type=int, default=0)
        self.parser.add_argument('--name', type=str, default=None)
        # key dimensions
        self.parser.add_argument('--emb_dim', type=int, default=256)
        self.parser.add_argument('--emb_cond_dim', type=int, default=256)
        self.parser.add_argument('--cond_dim', type=int, default=0) # no need to set
        # auto-encoder
        self.parser.add_argument('--AE_type', type=str, default='CVAE', choices=['CVAE'])
        self.parser.add_argument('--AE_loss', type=str, default='CE', choices=['BCE', 'BCEWithLogits', 'CE'])
        self.parser.add_argument('--alpha', type=float, default=1)
        self.parser.add_argument('--beta', type=float, default=0.0001)
        self.parser.add_argument('--detach_target', type=bool, default=True)
        self.parser.add_argument('--annealing', type=bool, default=True)
        self.parser.add_argument('--annealing_slope_length', type=int, default=100)
        self.parser.add_argument('--annealing_shape', type=str, default='cosine')
        self.parser.add_argument('--teacher_forcing', action='store_true', default=False)
        self.parser.add_argument('--guide_tensor', action='store_true', default=False)
        # auto-encoder architecture
        self.parser.add_argument('--embedding_type', type=str, default='None', choices=['None', 'GNN'])
        self.parser.add_argument('--encoder_type', type=str, default='MATRIX', choices=['None', 'MATRIX'])
        self.parser.add_argument('--decoder_type', type=str, default='MATRIX', choices=['MATRIX'])
        # gnn
        self.parser.add_argument('--num_layer', type=int, default=3)
        self.parser.add_argument('--JK', type=str, default='last')
        self.parser.add_argument('--readout', type=str, default='mean')
        self.parser.add_argument('--dropout_ratio', type=float, default=0.1)
        self.parser.add_argument('--gnn_type', type=str, default='gat')
        # training
        self.parser.add_argument('--augmentation', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=256)
        self.parser.add_argument('--epochs', type=int, default=500)
        self.parser.add_argument('--lr', type=float, default=0.0005)
        self.parser.add_argument('--lr_scale', type=float, default=1.0)
        self.parser.add_argument('--decay', type=float, default=0.00005)
        self.parser.add_argument('--gnn_lr_scale', type=float, default=1.0)
        self.parser.add_argument('--class_weight', type=str, default='enabled', choices=['enabled', 'disabled'])
        # finetuning
        self.parser.add_argument('--pretrained_file', type=str, default='None')
        self.parser.add_argument('--pretrained_time', type=str, default='None')
        # generation
        self.parser.add_argument('--correction', type=str, default='enabled', choices=['enabled', 'disabled'])
        self.parser.add_argument('--from_around_mol', type=str, default='enabled', choices=['enabled', 'disabled'])
        self.parser.add_argument('--from_around_mol_cond', type=str, default='enabled', choices=['enabled', 'disabled'])
        self.parser.add_argument('--from_training_space', type=str, default='None', choices=['enabled', 'disabled'])
        self.parser.add_argument('--from_guide', type=str, default='None') # no use for now
        # optimization
        self.parser.add_argument('--opt_strategy', type=str, default='at_random', choices=['at_random', 'around_target'])

    def setArgument(self, arguments=None):
        if arguments is not None:
            args = self.parser.parse_args(arguments)
        else:
            args = self.parser.parse_args()
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.gpu = 0 if torch.cuda.is_available() else -1
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms = True
        torch.set_deterministic_debug_mode("warn")
        torch.autograd.set_detect_anomaly(True)

        # argument dataset setup
        arguments_dataset = _dataset.dataset_args[args.file]
        args.file = arguments_dataset['file']
        args.smiles = arguments_dataset['smiles']
        args.time = arguments_dataset['time']
        args.task = arguments_dataset['task']
        args.ids = arguments_dataset['ids']
        args.splitting =arguments_dataset['splitting']
        args.graphtask = arguments_dataset['predictiontype']
        args.predictiontask = arguments_dataset['predictiontask']
        args.condition_dict = arguments_dataset['condition_dict']
        args.condition_extra = arguments_dataset['condition_extra'] if 'condition_extra' in arguments_dataset else None
        args.cond_dim = getConditionDim(args.condition_dict)+1

        args.folder_path = 'dataset/'+args.file
        if not os.path.exists(args.folder_path):
            os.makedirs(args.folder_path)

        args.max_atom_number = max_atom_number
        args.len_atom_type = len(atom_encoder_m) 
        args.len_bond_type = len(bond_encoder_m)
        args.matrix_size = matrix_size

        args.embedding_setting = {'type': args.embedding_type,
                                'num_layer': args.num_layer, 
                                'emb_dim': args.emb_dim, 
                                'emb_cond_dim': args.emb_cond_dim,
                                'JK': args.JK, 
                                'readout': args.readout, 
                                'dropout_ratio': args.dropout_ratio, 
                                'gnn_type': args.gnn_type}
        args.encoding_setting = {'type': args.encoder_type,
                                'emb_dim': args.emb_dim, 
                                'max_atom_number': args.max_atom_number, 
                                'len_atom_type': args.len_atom_type,
                                'len_bond_type': args.len_bond_type,
                                'matrix_size': args.matrix_size, 
                                'num_layer': args.num_layer, 
                                'dropout_ratio': args.dropout_ratio}
        args.decoding_setting = {'type': args.decoder_type,
                                'emb_dim': args.emb_dim, 
                                'emb_cond_dim': args.emb_cond_dim,
                                'cond_dim': args.cond_dim,
                                'max_atom_number': args.max_atom_number, 
                                'len_atom_type': args.len_atom_type,
                                'len_bond_type': args.len_bond_type,
                                'matrix_size': args.matrix_size, 
                                'num_layer': args.num_layer, 
                                'dropout_ratio': args.dropout_ratio, 
                                'teacher_forcing': args.teacher_forcing,
                                'guide_tensor': args.guide_tensor}
        print('args:', args)
        return args