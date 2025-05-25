import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import deepchem as dc
import selfies as sf
import os
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from catcvae.molgraph import AtomGraph, mol2matrix, num_node_features, num_edge_features
from catcvae.molgraph import augment_matrix, max_atom_number, atom_decoder_m, bond_decoder_m
from catcvae.condition import getOneHotCondition
from catcvae.utils import *
from multiprocessing import Pool

# Dataset getting and construction dataframe
def getDatasetFromFile(file, smiles, time, task, splitting=None, ids=None, condition=None):
    path = 'dataset/'+file+'.csv'

    if os.path.exists(path):
        df = pd.read_csv('dataset/'+file+'.csv')
        col_reactant = smiles['reactant']
        col_reagent = smiles['reagent']
        col_product = smiles['product']
        col_catalyst = smiles['catalyst']
        col_time = time
        col_condition = condition
        df_new = df[[col_reactant, col_reagent, col_product, col_catalyst, col_time, task]+col_condition].copy()
        x_col = ['X_reactant', 'X_reagent', 'X_product', 'X_catalyst', 'X_time']
        y_col = ['y']
        c_col = ['C_'+c for c in col_condition]
        df_new.columns = x_col+y_col+c_col
        if ids is not None:
            df_new['ids'] = list(df[ids])
        else:
            df_new['ids'] = list(df.index)
        if splitting is not None:
            df_new['s'] = df[splitting]
        else:
            df_new['s'] = np.zeros(len(df[task]))
        datasets = df_new
        datasets.fillna('', inplace=True)
    else:
        print("ERROR: file does not exist.")

    return datasets

# Custom data class
class MyData(Data):
    def __init__(self, 
                 x_reactant=None, edge_index_reactant=None, 
                 x_reagent=None, edge_index_reagent=None,
                 x_product=None, edge_index_product=None,
                 x_catalyst=None, edge_index_catalyst=None,
                 **kwargs):
        super().__init__()
        
        # reactant
        self.x_reactant = x_reactant if x_reactant is not None else None
        self.edge_index_reactant = edge_index_reactant if edge_index_reactant is not None else None
        # reagent
        self.x_reagent = x_reagent if x_reagent is not None else None
        self.edge_index_reagent = edge_index_reagent if edge_index_reagent is not None else None
        # product
        self.x_product = x_product if x_product is not None else None
        self.edge_index_product = edge_index_product if edge_index_product is not None else None
        # catalyst
        self.x_catalyst = x_catalyst if x_catalyst is not None else None
        self.edge_index_catalyst = edge_index_catalyst if edge_index_catalyst is not None else None
            
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_reactant':
            return self.x_reactant.size(0)
        elif key == 'edge_index_reagent':
            return self.x_reagent.size(0)
        elif key == 'edge_index_product':
            return self.x_product.size(0)
        elif key == 'edge_index_catalyst':
            return self.x_catalyst.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def getDataObject(args, d):
    # d = {
    # "X_reactant": ..., 
    # "X_reagent": ..., 
    # "X_product": ..., 
    # "X_catalyst": ...,
    # "y": 0.0, 
    # "ids": 0,
    # "C_X": ...
    # }
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # X_reactant
    try:
        mol = smiles_to_mol(d['X_reactant'], with_atom_index=False)
        sm = mol_to_smiles(mol)
        atom_graph = AtomGraph(sm)
        graph_size, node_index, node_features, edge_index, edge_features = \
            atom_graph.graph_size, atom_graph.node_index, atom_graph.node_features, atom_graph.edge_index, atom_graph.edge_features

        node_features_X_reactant = torch.tensor(np.array(node_features), dtype=torch.float).type(torch.FloatTensor)
        # data.edge_index: Graph connectivity with shape [2, num_edges]
        edge_index_X_reactant = torch.tensor(np.array(edge_index).T, dtype=torch.long).type(torch.LongTensor) \
            if graph_size != 1 and len(edge_index) != 0 else torch.tensor(np.array([[],[]])).type(torch.LongTensor)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_features_X_reactant = torch.tensor(np.array(edge_features), dtype=torch.float).type(torch.FloatTensor) \
            if graph_size != 1 and len(edge_features) != 0 else torch.Tensor(np.array([])).type(torch.FloatTensor)
    except Exception as e:
        print('[ERROR] getDataObject (X_reactant): ', d['ids'], d['X_reactant'], e)
        return None

    # X_reagent
    try:
        mol = smiles_to_mol(d['X_reagent'], with_atom_index=False)
        sm = mol_to_smiles(mol)
        atom_graph = AtomGraph(sm)
        graph_size, node_index, node_features, edge_index, edge_features = \
            atom_graph.graph_size, atom_graph.node_index, atom_graph.node_features, atom_graph.edge_index, atom_graph.edge_features

        node_features_X_reagent = torch.tensor(np.array(node_features), dtype=torch.float).type(torch.FloatTensor) \
            if graph_size != 0 else torch.Tensor(np.zeros((1, num_node_features))).type(torch.FloatTensor)
        # data.edge_index: Graph connectivity with shape [2, num_edges]
        edge_index_X_reagent = torch.tensor(np.array(edge_index).T, dtype=torch.long).type(torch.LongTensor) \
            if graph_size != 1 and len(edge_index) != 0 else torch.tensor(np.array([[],[]])).type(torch.LongTensor)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_features_X_reagent = torch.tensor(np.array(edge_features), dtype=torch.float).type(torch.FloatTensor) \
            if graph_size != 1 and len(edge_features) != 0 else torch.Tensor(np.array([])).type(torch.FloatTensor)
    except Exception as e:
        print('[ERROR] getDataObject (X_reagent): ', d['ids'], d['X_reagent'], e)
        return None

    # X_product
    try:
        mol = smiles_to_mol(d['X_product'], with_atom_index=False)
        sm = mol_to_smiles(mol)
        atom_graph = AtomGraph(sm)
        graph_size, node_index, node_features, edge_index, edge_features = \
            atom_graph.graph_size, atom_graph.node_index, atom_graph.node_features, atom_graph.edge_index, atom_graph.edge_features

        node_features_X_product = torch.tensor(np.array(node_features), dtype=torch.float).type(torch.FloatTensor)
        # data.edge_index: Graph connectivity with shape [2, num_edges]
        edge_index_X_product = torch.tensor(np.array(edge_index).T, dtype=torch.long).type(torch.LongTensor) \
            if graph_size != 1 and len(edge_index) != 0 else torch.tensor(np.array([[],[]])).type(torch.LongTensor)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_features_X_product = torch.tensor(np.array(edge_features), dtype=torch.float).type(torch.FloatTensor) \
            if graph_size != 1 and len(edge_features) != 0 else torch.Tensor(np.array([])).type(torch.FloatTensor)
    except Exception as e:
        print('[ERROR] getDataObject (X_product): ', d['ids'], d['X_product'], e)
        return None

    # X_catalyst
    try:
        mol = smiles_to_mol(d['X_catalyst'], with_atom_index=False)
        sm = mol_to_smiles(mol)
        atom_graph = AtomGraph(sm)
        graph_size, node_index, node_features, edge_index, edge_features = \
            atom_graph.graph_size, atom_graph.node_index, atom_graph.node_features, atom_graph.edge_index, atom_graph.edge_features

        node_features_X_catalyst = torch.tensor(np.array(node_features), dtype=torch.float).type(torch.FloatTensor) \
            if graph_size != 0 else torch.Tensor(np.zeros((1, num_node_features))).type(torch.FloatTensor)
        # data.edge_index: Graph connectivity with shape [2, num_edges]
        edge_index_X_catalyst = torch.tensor(np.array(edge_index).T, dtype=torch.long).type(torch.LongTensor) \
            if graph_size != 1 and len(edge_index) != 0 else torch.tensor(np.array([[],[]])).type(torch.LongTensor)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_features_X_catalyst = torch.tensor(np.array(edge_features), dtype=torch.float).type(torch.FloatTensor) \
            if graph_size != 1 and len(edge_features) != 0 else torch.Tensor(np.array([])).type(torch.FloatTensor)
    except Exception as e:
        print('[ERROR] getDataObject (X_catalyst): ', d['ids'], d['X_catalyst'], e)
        return None

    # smiles (reactant>reagent.catalyst>product)
    smiles = '>'.join(filter(None, [d['X_reactant'], '.'.join(filter(None, [d['X_reagent'], d['X_catalyst']])), d['X_product']])) 
    
    # catalyst matrix
    mol_cat = smiles_to_mol(d['X_catalyst'], with_atom_index=False)
    matrix_catalyst = torch.tensor(np.array(mol2matrix(mol_cat)), dtype=torch.float).type(torch.FloatTensor)
    
    # condition
    condition_list = getOneHotCondition({c: d['C_'+c] for c in args.condition_dict.keys()}, args.condition_dict)
    condition_list = condition_list+[float(d['X_time'])]
    condition = torch.tensor(np.array([condition_list]), dtype=torch.float).type(torch.FloatTensor)

    dobj = MyData(#x=node_features, edge_index=edge_index, edge_attr=edge_features, num_nodes=graph_size,
                    x_reactant=node_features_X_reactant, edge_index_reactant=edge_index_X_reactant, edge_attr_reactant=edge_features_X_reactant,
                    x_reagent=node_features_X_reagent, edge_index_reagent=edge_index_X_reagent, edge_attr_reagent=edge_features_X_reagent,
                    x_product=node_features_X_product, edge_index_product=edge_index_X_product, edge_attr_product=edge_features_X_product,
                    x_catalyst=node_features_X_catalyst, edge_index_catalyst=edge_index_X_catalyst, edge_attr_catalyst=edge_features_X_catalyst,
                    y=torch.tensor([d['y']], dtype=torch.float), id=str(d['ids']), c=condition, smiles=smiles, time_h=d['X_time'], graph_size_cat=graph_size,
                    smiles_reactant=d['X_reactant'], smiles_reagent=d['X_reagent'], smiles_product=d['X_product'], smiles_catalyst=d['X_catalyst'],
                    matrix_catalyst=matrix_catalyst)
    return dobj


# def getDataObject_wrapper(args):
#     return getDataObject(*args)


def getDatasetObject(args, datasets_df):
    folder_path = 'dataset/'+args.file
    path = folder_path+'/graph.pickle'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_dobj = pickle.load(handle)
    else:
        # print('Constructing dataset object with', os.cpu_count(), 'cores...')
        # value = [(args, d) for idx, d in datasets_df.iterrows()]
        # with Pool(processes=os.cpu_count()) as pool:
        #     datasets_dobj = list(tqdm(pool.imap(getDataObject_wrapper, value), total=len(value)))
        #     pool.close()
        #     pool.join()
        datasets_dobj = []
        for idx, d in tqdm(datasets_df.iterrows()):
            try:
                dobj = getDataObject(args, d)
                datasets_dobj.append(dobj)
            except Exception as e:
                print('[ERROR] getDatasetObject: ', d['ids'], e)
        
        # write dataset to pickle 
        with open(path, 'wb') as handle:
            pickle.dump(datasets_dobj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return datasets_dobj

# data splitting for pre-training dataset
def getDatasetSplitting(args, datasets_df=None, datasets_dobj=None, augmentation=0):
    folder_path = 'dataset/'+args.file
    path_train = folder_path+'/datasets_dobj_train_'+str(args.seed)+'.pkl'
    path_val = folder_path+'/datasets_dobj_val_'+str(args.seed)+'.pkl'
    path_test = folder_path+'/datasets_dobj_test_'+str(args.seed)+'.pkl'

    if os.path.exists(path_train):
        with open(path_train, 'rb') as handle:
            datasets_dobj_train = pickle.load(handle)
        with open(path_val, 'rb') as handle:
            datasets_dobj_val = pickle.load(handle)
        with open(path_test, 'rb') as handle:
            datasets_dobj_test = pickle.load(handle)

    else:
        datasets = dc.data.DiskDataset.from_numpy(X=datasets_df['ids'].values, y=datasets_df['y'].values, ids=datasets_df['ids'].values)
        datasets_size = len(datasets)
        # splitting for pre-training 95/5/5
        splitter_s = dc.splits.RandomSplitter()
        datasets_trainval, datasets_test = splitter_s.train_test_split(dataset=datasets, seed=args.seed, frac_train=0.95)
        datasets_frac = (0.9*datasets_size)/len(datasets_trainval)
        datasets_train, datasets_val = splitter_s.train_test_split(dataset=datasets_trainval, seed=args.seed, frac_train=datasets_frac)
        
        datasets_dobj_train = []
        datasets_dobj_val = []
        datasets_dobj_test = []
        for d in tqdm(datasets_dobj):
            if 'id' in d.keys():
                if d.id in datasets_train.ids or d.id in datasets_train.ids.astype(str):
                    datasets_dobj_train.append(d)
                    # data augmentation only train set (node order shuffling)
                    if augmentation!=0:
                        if d.graph_size_cat > 1:
                            for _ in range(augmentation):
                                matrix_catalyst_augmented = augment_matrix(d.matrix_catalyst.detach().numpy(), max_atom_number, atom_decoder_m, bond_decoder_m)
                                d_augmented = copy.deepcopy(d)
                                d_augmented.matrix_catalyst = torch.from_numpy(matrix_catalyst_augmented).type(torch.FloatTensor)
                                d_augmented.id = str(d.id)+"_aug_"+str(_)
                                datasets_dobj_train.append(d_augmented)
                if d.id in datasets_val.ids or d.id in datasets_val.ids.astype(str):
                    datasets_dobj_val.append(d)
                if d.id in datasets_test.ids or d.id in datasets_test.ids.astype(str):
                    datasets_dobj_test.append(d)
        
        # write dataset to pickle 
        with open(path_train, 'wb') as handle:
            pickle.dump(datasets_dobj_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path_val, 'wb') as handle:
            pickle.dump(datasets_dobj_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path_test, 'wb') as handle:
            pickle.dump(datasets_dobj_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return datasets_dobj_train, datasets_dobj_val, datasets_dobj_test

# data splitting for fine-tuning dataset
def getDatasetSplittingFinetune(args, datasets_df=None, datasets_dobj=None, augmentation=0):
    folder_path = 'dataset/'+args.file
    path_train = folder_path+'/datasets_dobj_train_'+str(args.seed)+'.pkl'
    path_val = folder_path+'/datasets_dobj_val_'+str(args.seed)+'.pkl'
    path_test = folder_path+'/datasets_dobj_test_'+str(args.seed)+'.pkl'

    if os.path.exists(path_train):
        with open(path_train, 'rb') as handle:
            datasets_dobj_train = pickle.load(handle)
        with open(path_val, 'rb') as handle:
            datasets_dobj_val = pickle.load(handle)
        with open(path_test, 'rb') as handle:
            datasets_dobj_test = pickle.load(handle)

    else:
        # if splitting is not specified
        if args.splitting is None:
            datasets = dc.data.DiskDataset.from_numpy(X=datasets_df['ids'].values, y=datasets_df['y'].values, ids=datasets_df['ids'].values)
            datasets_size = len(datasets)
            
            # splitting for fine-tuning (default: 70/10/20)
            splitter_s = dc.splits.RandomSplitter()
            if 'suzuki' in args.file or 'ps_ddG' in args.file: #80/10/10
                datasets_trainval, datasets_test = splitter_s.train_test_split(dataset=datasets, seed=args.seed, frac_train=0.90)
                datasets_frac = (0.8*datasets_size)/len(datasets_trainval)
                datasets_train, datasets_val = splitter_s.train_test_split(dataset=datasets_trainval, seed=args.seed, frac_train=datasets_frac)
                # datasets_train, datasets_val, datasets_test = splitter_s.train_valid_test_split(dataset=datasets, seed=args.seed)
            elif 'ru' in args.file: #60/10/30
                datasets_trainval, datasets_test = splitter_s.train_test_split(dataset=datasets, seed=args.seed, frac_train=0.70)
                datasets_frac = (0.6*datasets_size)/len(datasets_trainval)
                datasets_train, datasets_val = splitter_s.train_test_split(dataset=datasets_trainval, seed=args.seed, frac_train=datasets_frac)
                # datasets_train, datasets_val, datasets_test = splitter_s.train_valid_test_split(dataset=datasets, seed=args.seed)
            else: #70/10/20
                datasets_trainval, datasets_test = splitter_s.train_test_split(dataset=datasets, seed=args.seed, frac_train=0.80)
                # if large dataset (more than 30 samples), split into train and val
                if datasets_size > 30:
                    datasets_frac = (0.7*datasets_size)/len(datasets_trainval)
                    datasets_train, datasets_val = splitter_s.train_test_split(dataset=datasets_trainval, seed=args.seed, frac_train=datasets_frac)
                    # datasets_train, datasets_val, datasets_test = splitter_s.train_valid_test_split(dataset=datasets, seed=args.seed)
                # if small dataset (less than 30 samples), use train and test as val
                else:
                    datasets_train = datasets_trainval
                    datasets_val = datasets_test

            datasets_dobj_train = []
            datasets_dobj_val = []
            datasets_dobj_test = []
            for d in tqdm(datasets_dobj):
                if 'id' in d.keys():
                    if d.id in datasets_train.ids or d.id in datasets_train.ids.astype(str):
                        datasets_dobj_train.append(d)
                        # data augmentation only train set (node order shuffling)
                        if augmentation!=0:
                            if d.graph_size_cat > 1:
                                for _ in range(augmentation):
                                    matrix_catalyst_augmented = augment_matrix(d.matrix_catalyst.detach().numpy(), max_atom_number, atom_decoder_m, bond_decoder_m)
                                    d_augmented = copy.deepcopy(d)
                                    d_augmented.matrix_catalyst = torch.from_numpy(matrix_catalyst_augmented).type(torch.FloatTensor)
                                    d_augmented.id = str(d.id)+"_aug_"+str(_)
                                    datasets_dobj_train.append(d_augmented)
                    if d.id in datasets_val.ids or d.id in datasets_val.ids.astype(str):
                        datasets_dobj_val.append(d)
                    if d.id in datasets_test.ids or d.id in datasets_test.ids.astype(str):
                        datasets_dobj_test.append(d)
        # if splitting is specified
        else:
            datasets_dobj_train = []
            datasets_dobj_val = []
            datasets_dobj_test = []
            assert len(datasets_dobj) == len(datasets_df['s']), 'Length of datasets_dobj and datasets_df[s] must be the same.'
            for d, s in tqdm(zip(datasets_dobj, datasets_df['s'])):
                if 'id' in d.keys():
                    if s.lower() == 'train':
                        datasets_dobj_train.append(d)
                    elif s.lower() == 'val' or s.lower() == 'validate':
                        datasets_dobj_val.append(d)
                    elif s.lower() == 'test':
                        datasets_dobj_test.append(d)
            # if no val set is specified, split train set into train and val (defult: 10% of train set)
            if len(datasets_dobj_val) == 0:
                if 'cpa_asymmetric' in args.file:
                    if 'FullCV' in args.file:
                        # random some 10% of train to val
                        datasets_frac = 0.1
                    else:
                        # random some 30% of train to val
                        datasets_frac = 0.3
                else:
                    # random some 10% of train to val
                    datasets_frac = 0.1
                random_indices = np.random.choice(len(datasets_dobj_train), int(datasets_frac * len(datasets_dobj_train)), replace=False)
                datasets_dobj_val += [datasets_dobj_train[i] for i in random_indices]
                datasets_dobj_train = [d for i, d in enumerate(datasets_dobj_train) if i not in random_indices]
            
            datasets_dobj_train_final = []
            for d in datasets_dobj_train:
                datasets_dobj_train_final.append(d)
                # data augmentation only train set (node order shuffling)
                if augmentation!=0:
                    if d.graph_size_cat > 1:
                        for _ in range(augmentation):
                            matrix_catalyst_augmented = augment_matrix(d.matrix_catalyst.detach().numpy(), max_atom_number, atom_decoder_m, bond_decoder_m)
                            d_augmented = copy.deepcopy(d)
                            d_augmented.matrix_catalyst = torch.from_numpy(matrix_catalyst_augmented).type(torch.FloatTensor)
                            d_augmented.id = str(d.id)+"_aug_"+str(_)
                            datasets_dobj_train_final.append(d_augmented)
            datasets_dobj_train = datasets_dobj_train_final
        
        # write dataset to pickle 
        with open(path_train, 'wb') as handle:
            pickle.dump(datasets_dobj_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path_val, 'wb') as handle:
            pickle.dump(datasets_dobj_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path_test, 'wb') as handle:
            pickle.dump(datasets_dobj_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # write dataset splitting to csv
    def create_df(datasets, split):
        return pd.DataFrame({
            'ids': [d.id for d in datasets],
            's': split,
            'smiles': [d.smiles for d in datasets],
            'smiles_reactant': [d.smiles_reactant for d in datasets],
            'smiles_reagent': [d.smiles_reagent for d in datasets],
            'smiles_product': [d.smiles_product for d in datasets],
            'smiles_catalyst': [d.smiles_catalyst for d in datasets],
            'time_h': [d.time_h for d in datasets],
            'y': [d.y.item() for d in datasets]
        })
    train_df = create_df(datasets_dobj_train, 'train')
    val_df = create_df(datasets_dobj_val, 'val')
    test_df = create_df(datasets_dobj_test, 'test')
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(folder_path+'/datasets_dobj_split_'+str(args.seed)+'.csv', index=False)

    return datasets_dobj_train, datasets_dobj_val, datasets_dobj_test

def getDataLoader(args, datasets_dobj_train, datasets_dobj_val, datasets_dobj_test):
    loader_train = DataLoader(datasets_dobj_train, batch_size=args.batch_size, shuffle=True, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
    loader_val = DataLoader(datasets_dobj_val, batch_size=args.batch_size, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
    loader_test = DataLoader(datasets_dobj_test, batch_size=args.batch_size, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
    return loader_train, loader_val, loader_test