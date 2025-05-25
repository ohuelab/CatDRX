# Library
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# catcvae
from dataset import _dataset
from catcvae.setup import ModelArgumentParser
from catcvae.molgraph import atom_encoder_m, bond_encoder_m, atom_decoder_m, bond_decoder_m, max_atom_number, matrix_size, matrix2mol
from catcvae.dataset import getDatasetFromFile, getDatasetObject, getDatasetSplitting, getDataLoader, getDataObject, getDatasetSplittingFinetune
from catcvae.condition import getConditionDim, getOneHotCondition
from catcvae.classweight import getClassWeight
from catcvae.loss import VAELoss, Annealer, recon_loss_fn, cosine_similarity
from catcvae.ae import CVAE
from catcvae.ae import latent_space_quality, sample_latent_space
from catcvae.training import save_model, save_loss, save_report, save_model_latest, save_model_latest_temp, write_continue_training
from catcvae.prediction import NN, NN_TASK
from catcvae.latent import embed, save_latent

# Main
if __name__ == '__main__':
    start_time_run = time.time()
    
    # argument setup
    parser = ModelArgumentParser()
    args = parser.setArgument()

    datasets_df = getDatasetFromFile(args.file, args.smiles, args.time, args.task, args.splitting, args.ids, list(args.condition_dict.keys()))
    print('datasets:', len(datasets_df))

    # finetune dataset
    path_train = args.folder_path+'/datasets_dobj_train_'+str(args.seed)+'.pkl'
    if os.path.exists(path_train):
        # get datasets splitting
        datasets_dobj_train, datasets_dobj_val, datasets_dobj_test = getDatasetSplittingFinetune(args, datasets_df=None, datasets_dobj=None, augmentation=args.augmentation)
        print('datasets_dobj_train:', len(datasets_dobj_train))
        print('datasets_dobj_val:', len(datasets_dobj_val))
        print('datasets_dobj_test:', len(datasets_dobj_test))

    loader_train, loader_val, loader_test = getDataLoader(args, datasets_dobj_train, datasets_dobj_val, datasets_dobj_test)

    # setup generative model
    AE = CVAE(embedding_setting=args.embedding_setting,
                encoding_setting=args.encoding_setting, 
                decoding_setting=args.decoding_setting,
                emb_dim=args.emb_dim,
                emb_cond_dim=args.emb_cond_dim,
                cond_dim=args.cond_dim, 
                device=args.device).to(args.device)
    # setup predictive model
    if args.predictiontask == 'yield':
        NN_PREDICTION = NN(in_dim=args.emb_dim+(3*args.emb_cond_dim)+args.cond_dim, out_dim_class=1).to(args.device)
    elif args.predictiontask == 'others':
        NN_PREDICTION = NN_TASK(in_dim=args.emb_dim+(3*args.emb_cond_dim)+args.cond_dim, out_dim_class=1).to(args.device)
    print(NN_PREDICTION)

    # finetuned dataset folder 
    file_trained = args.pretrained_file
    seed_trained = 0
    time_trained = args.pretrained_time
    output_model_dir_trained = 'dataset/'+file_trained+'/output'+'_'+str(seed_trained)+'_'+time_trained
    epoch_selected = None
    AE.load_state_dict(torch.load(output_model_dir_trained + '/model_ae.pth', map_location=torch.device(args.device)))
    NN_PREDICTION.load_state_dict(torch.load(output_model_dir_trained + '/model_nn.pth', map_location=torch.device(args.device)))

    AE.eval()
    NN_PREDICTION.eval()

    # run embedding
    mol_latent_train, mol_embedding_train, y_true_train, y_pred_train, ids_train, c_train = embed(loader_train, AE, NN_PREDICTION, device=args.device)
    mol_latent_val, mol_embedding_val, y_true_val, y_pred_val, ids_val, c_val = embed(loader_val, AE, NN_PREDICTION, device=args.device)
    mol_latent_test, mol_embedding_test, y_true_test, y_pred_test, ids_test, c_test = embed(loader_test, AE, NN_PREDICTION, device=args.device)

    # min max for each dimension in mol_embedding
    min_val = np.min(mol_embedding_train, axis=0)
    max_val = np.max(mol_embedding_train, axis=0)
    dimension_training = list(zip(np.floor(min_val), np.ceil(max_val)))
    print('dimension:', dimension_training)
    print('len(dimension):', len(dimension_training))

    # specific train index if any
    # specific_train_index = []
    # for i, mol in enumerate(datasets_dobj_train):
    #     if mol.smiles_catalyst is not None:
    #         if 'Pd' in mol.smiles_catalyst:
    #             specific_train_index.append(i)

    def random_dataset(index=None):
        AE.eval()
        NN_PREDICTION.eval()
        # t = specific_train_index[index] if index is not None else random.choice(specific_train_index)
        t = index if index is not None else random.randint(0, len(datasets_dobj_train)-1)
        t = t%len(datasets_dobj_train) if t >= len(datasets_dobj_train) else t
        mol_t = datasets_dobj_train[t]
        loader_mol_t = DataLoader([mol_t], batch_size=1, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
        mol_latent_t, mol_embedding_t, y_true_t, y_pred_t, ids_t, c_t = embed(loader_mol_t, AE, NN_PREDICTION, device=args.device)
        mol_embedding_t = mol_embedding_t.squeeze()
        c_detail = {'smiles_reactant': mol_t.smiles_reactant, 
                  'smiles_reagent': mol_t.smiles_reagent, 
                  'smiles_product': mol_t.smiles_product, 
                  'smiles_catalyst': mol_t.smiles_catalyst,
                  'time_h': mol_t.time_h,
                  'c': mol_t.c,
                  'y': mol_t.y.item()}
        return mol_embedding_t, c_t, c_detail

    from rdkit import DataStructs

    def get_fingerprint_dictionary(smiles_list):
        result = {}
        for smiles in tqdm(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)
                result[smiles] = fp
        return result

    def similarity(a, b, radius=2, dictionary=None):
        if a is None or b is None: 
            return 0.0
        if dictionary and a in dictionary and b in dictionary:
            fp1 = dictionary[a]
            fp2 = dictionary[b]
        else:
            amol = Chem.MolFromSmiles(a)
            bmol = Chem.MolFromSmiles(b)
            if amol is None or bmol is None:
                # print(a, b)
                return 0.0
            fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, radius=radius, nBits=2048, useChirality=False)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, radius=radius, nBits=2048, useChirality=False)
        return DataStructs.TanimotoSimilarity(fp1, fp2) 
    
    def similarity_to_nearest_neighbor(smiles_list, ref_list, radius=2, dictionary=None):
        similarity_list = []
        for i, a in tqdm(enumerate(smiles_list)):
            max_similarity = 0
            for b in ref_list:
                sim = similarity(a, b, radius=radius, dictionary=dictionary)
                if sim > max_similarity:
                    max_similarity = sim
            similarity_list.append(max_similarity)
        return np.mean(similarity_list), np.std(similarity_list)
    
    # get fingerprint dictionary from dataset
    training_smiles = [mol.smiles_catalyst for mol in datasets_dobj_train]
    fingerprint_dict = get_fingerprint_dictionary([mol.smiles_catalyst for mol in datasets_dobj_train])

    import numpy as np
    import matplotlib.pyplot as plt
    from rdkit.Chem import Descriptors
    from rdkit.Chem import RDConfig
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from skopt import gp_minimize

    # search_strategy = 'at_random'
    # search_strategy = 'around_target'
    search_strategy = args.opt_strategy
    around_target_dimension = 5.0

    objective_value = 100
    error_value = 100
    if 'cc' in args.file:
        objective_value = -27.55
        error_value = 100

    # objective_setup = {'objective_value': objective_value,
    #                    'error_value': error_value,
    #                    'chemical_rule': True,
    #                    'number_of_fragment': 3,
    #                 #    'similarity': True,
    #                 #    'sa_score': True
    #                    }
    objective_setup = {'objective_value': objective_value,
                       'error_value': error_value,
                       'number_of_fragment': 4,
                    #    'sa_score': True,
                       }

    def objective_function(z, c_t=None, c_detail=None, mol_latent_starting=None):

        if search_strategy == 'around_target':
            z = mol_latent_starting + torch.tensor([np.array(z)], dtype=torch.float).type(torch.FloatTensor).to(args.device)
        elif search_strategy == 'at_random':
            z = torch.tensor([np.array(z)], dtype=torch.float).type(torch.FloatTensor).to(args.device)

        try:
            out_decoded = AE.decode(z, c_t) # decode cvae
            out_matrix = out_decoded.reshape(max_atom_number, matrix_size//max_atom_number)
            mol_catalyst = matrix2mol(out_matrix, print_result=False)
            # display(mol_catalyst)

            mol_catalyst.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol_catalyst)
            smiles_catalyst = Chem.MolToSmiles(mol_catalyst)
            d = {
                "X_reactant": c_detail['smiles_reactant'],
                "X_reagent": c_detail['smiles_reagent'],
                "X_product": c_detail['smiles_product'],
                "X_catalyst": smiles_catalyst,
                "X_time": c_detail['time_h'],
                "y": 0.0, 
                "ids": 0,
            }
            try:
                d['C_'+list(args.condition_dict.keys())[0]] = Descriptors.ExactMolWt(mol_catalyst)
                # d['C_mw_ligand'] = Descriptors.ExactMolWt(mol_catalyst)
            except:
                d['C_'+list(args.condition_dict.keys())[0]] = 200
                # d['C_mw_ligand'] = 200

            sample_mol_t = getDataObject(args, d)
            loader_mol_t = DataLoader([sample_mol_t], batch_size=1, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
            sample_mol_latent_t, sample_mol_embedding_t, sample_y_true_t, sample_y_pred_t, sample_ids_t, sample_c_t = embed(loader_mol_t, AE, NN_PREDICTION, device=args.device)
            # display(Chem.MolFromSmiles(sample_mol_t.smiles_catalyst))
            # print(sample_y_pred_t)

            if 'objective_value' in objective_setup:
                objective = np.abs(sample_y_pred_t.item() - objective_setup['objective_value'])
            if 'sa_score' in objective_setup:
                try:
                    sa = sascorer.calculateScore(mol_catalyst)
                except:
                    sa = 7.0
                if sa >= 6.0:
                    objective += 50
                    # objective = 0
            if 'number_of_fragment' in objective_setup:
                rs = Chem.GetMolFrags(mol_catalyst, asMols=True)
                if 'cc' in args.file:
                    # metal = ['Pd', 'Ni', 'Cu', 'Ag', 'Au', 'Pt']
                    metal = ['Pd']
                    m_validity_3frag = 0
                    mol_frag = mol_catalyst
                    try:
                        rs = Chem.GetMolFrags(mol_frag, asMols=True)
                        # if len(rs) == 3 and any(m in s for m in metal):
                        if len(rs) == 3:
                            flag_metal = False
                            for r in rs:
                                # get number of atoms
                                if r.GetNumAtoms() == 1:
                                    # get atom symbol
                                    atom = r.GetAtoms()[0]
                                    if atom.GetSymbol() in metal:
                                        flag_metal = True
                                        break
                            if not flag_metal:
                                objective += 100
                        else:
                            objective += 100
                    except:
                        objective += 100
                else:
                    if len(rs) != objective_setup['number_of_fragment']:
                        objective += 100
                        # objective = 0
            if 'chemical_rule' in objective_setup:
                if 'cc' in args.file:
                    rs = Chem.GetMolFrags(mol_catalyst, asMols=True)
                    if len(rs) == 3:
                        for r in rs:
                            if r.GetNumAtoms() != 1:
                                # check atom P or atom N in molecule
                                has_P = False
                                has_N = False
                                has_O = False
                                for atom in r.GetAtoms():
                                    if atom.GetSymbol() == 'P':
                                        has_P = True
                                    if atom.GetSymbol() == 'N':
                                        has_N = True
                                    if atom.GetSymbol() == 'O':
                                        has_O = True
                                if not has_P and not has_N and not has_O:
                                    objective += 300
                                    # objective = 0
                                # check neightbor of atom P is exactly 3
                                if has_P:
                                    for atom in r.GetAtoms():
                                        if atom.GetSymbol() == 'P':
                                            if atom.GetDegree() != 3:
                                                objective += 100
                                                # objective = 0
                                                break
                                # check P have three neighbor
                                if has_N:
                                    for atom in r.GetAtoms():
                                        if atom.GetSymbol() == 'N':
                                            if atom.GetDegree() > 3:
                                                objective += 100
                                                # objective = 0
                                                break
                                # check O have three neighbor
                                if has_O:
                                    for atom in r.GetAtoms():
                                        if atom.GetSymbol() == 'O':
                                            if atom.GetDegree() > 2:
                                                objective += 100
                                                # objective = 0
                                                break
                                # check contain 3-member ring
                                ssr = Chem.GetSymmSSSR(r)
                                has_3member_ring = any(len(ring) < 5 for ring in ssr)
                                if has_3member_ring:
                                    objective += 100
                                    # objective = 0
            if 'similarity' in objective_setup:
                sim = similarity_to_nearest_neighbor([smiles_catalyst], training_smiles, dictionary=fingerprint_dict)
                # if sim[0] > 0.5:
                #     objective -= 20
                # if sim[0] < 0.3:
                #     objective += 5
                #     objective = 0
                if sim[0] < 0.2:
                    objective += 50
                    # objective = 0
        
            return_value = objective

            return return_value

        except Exception as e:
            print(e)
            return objective_setup['error_value']
        
    # from tqdm import tqdm_notebook as tqdm
    from tqdm.notebook import tqdm
    class tqdm_skopt(object):
        def __init__(self, **kwargs):
            self._bar = tqdm(**kwargs)
        def __call__(self, res):
            self._bar.update()

    starting_mol_list = []
    starting_mol_list_y = []
    save_time = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_model_dir_trained + '/optimize_smiles_'+save_time, exist_ok=True)

    for round in range(2):
    # for round in [46, 196, 386, 546, 866, 936]:

        mol_latent, cond_random, c_detail = random_dataset(index=round)
        c_t = torch.tensor(np.array(cond_random), dtype=torch.float).type(torch.FloatTensor).to(args.device)
        if search_strategy == 'around_target':
            print('Starting molecule:', c_detail)
            starting_mol_list.append(c_detail['smiles_catalyst'])
            starting_mol_list_y.append(c_detail['y'])
            mol_latent_starting = torch.tensor(mol_latent, device=args.device)
        
        elif search_strategy == 'at_random':
            mol_latent_starting = None
        
        def objective_function_wrapping(z):
            return objective_function(z, c_t, c_detail, mol_latent_starting)
            
        if search_strategy == 'around_target':
            dimension = around_target_dimension
            dimension_bounds = [(-dimension, dimension)]*args.emb_dim
        elif search_strategy == 'at_random':
            dimension_bounds = dimension_training
        
        n_calls  = 100
        n_initial_points = 50
        noise = "gaussian"

        res = gp_minimize(objective_function_wrapping,                # the function to minimize
                        dimension_bounds,                             # the bounds on each dimension of x
                        acq_func="gp_hedge",                          # the acquisition function
                        n_calls=n_calls,                              # the number of evaluations of f
                        n_initial_points=n_initial_points,            # the number of random initialization points
                        noise=noise,                                  # the noise level (optional)
                        # random_state=args.seed,                     # the random seed
                        random_state=round,
                        # random_state=random.randint(0, 10000),      # the random seed
                        callback=[tqdm_skopt(total=n_calls)],         # the callback function
                        n_jobs=-1                                     # the number of cores to use in parallel
                        )         
        
        # display top molecules with predicted value from res
        top = 10
        func_vals_top = np.argsort(-res.func_vals)[-top:]
        top_molecules = []
        top_molecules_smiles = []
        top_molecules_y = []
        top_molecules_prediction = []
        top_molecules_objective = []

        for i in range(top):
            # print("TOP", top-i)
            min_x = torch.tensor([np.array(res.x_iters[func_vals_top[i]])], dtype=torch.float).type(torch.FloatTensor).to(args.device)
            
            if search_strategy == 'around_target':
                min_x = mol_latent_starting + min_x
            elif search_strategy == 'at_random':
                min_x = min_x
            
            out_decoded = AE.decode(min_x, c_t) # decode cvae
            # y_prediction = NN_PREDICTION(min_x, c_t)
            out_matrix = out_decoded.reshape(max_atom_number, matrix_size//max_atom_number)
            mol_catalyst = matrix2mol(out_matrix, print_result=False)
            
            mol_catalyst.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol_catalyst)
            smiles_catalyst = Chem.MolToSmiles(mol_catalyst)
            smiles_catalyst = Chem.CanonSmiles(smiles_catalyst, useChiral=False)
            # dummy object
            d = {
                "X_reactant": c_detail['smiles_reactant'],
                "X_reagent": c_detail['smiles_reagent'],
                "X_product": c_detail['smiles_product'],
                "X_catalyst": smiles_catalyst,
                "X_time": c_detail['time_h'],
                "y": 0.0, 
                "ids": 0,
            }
            try:
                d['C_'+list(args.condition_dict.keys())[0]] = Descriptors.ExactMolWt(mol_catalyst)
                # d['C_mw_ligand'] = Descriptors.ExactMolWt(mol_catalyst)
            except:
                d['C_'+list(args.condition_dict.keys())[0]] = 200
                # d['C_mw_ligand'] = 200

            sample_mol_t = getDataObject(args, d)
            loader_mol_t = DataLoader([sample_mol_t], batch_size=1, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
            sample_mol_latent_t, sample_mol_embedding_t, sample_y_true_t, sample_y_pred_t, sample_ids_t, sample_c_t = embed(loader_mol_t, AE, NN_PREDICTION, device=args.device)
                
            # print(out_smiles)
            # print(res.func_vals[func_vals_top[i]])
            # display(Chem.MolFromSmiles(out_smiles))
            top_molecules.append(mol_catalyst)
            top_molecules_smiles.append(Chem.MolToSmiles(mol_catalyst))
            top_molecules_y.append(f'{res.func_vals[func_vals_top[i]]:.2f} vs {sample_y_pred_t.item():.2f}')
            top_molecules_prediction.append(sample_y_pred_t.item())
            top_molecules_objective.append(res.func_vals[func_vals_top[i]])

        # display_molecule(top_molecules, title=range(1,top+1), texts=top_molecules_y)
        try:
            from rdkit.Chem import Draw
            if len(top_molecules) > 0:
                top_molecules = [mol for mol in top_molecules if mol is not None]
                img = Draw.MolsToGridImage(top_molecules, molsPerRow=5, subImgSize=(200,200), legends=top_molecules_y)
                img.save(output_model_dir_trained + '/optimize_smiles_'+save_time+'/'+str(round)+'.png')
        except Exception as e:
            print(e)
            pass

        with open(output_model_dir_trained + '/optimize_smiles_'+save_time+'/smiles.txt', 'a') as f:
            for i in range(len(top_molecules)):
                if search_strategy == 'around_target':
                    f.write(f'{round} {top_molecules_smiles[i]} {top_molecules_prediction[i]:.2f} {top_molecules_objective[i]:.2f} {c_detail["smiles_catalyst"]} {c_detail["y"]}\n')
                elif search_strategy == 'at_random':
                    f.write(f'{round} {top_molecules_smiles[i]} {top_molecules_prediction[i]:.2f} {top_molecules_objective[i]:.2f}\n')
     

        for i, mol in enumerate(top_molecules):
            print(top_molecules_smiles[i])
            print(top_molecules_y[i])
            # display(mol)

    if search_strategy == 'around_target':
        try:
            from rdkit.Chem import Draw
            if len(starting_mol_list) > 0:
                starting_mol = [Chem.MolFromSmiles(s) for s in starting_mol_list]
                starting_mol_y = [f'{y:.2f}' for y in starting_mol_list_y]
                img = Draw.MolsToGridImage(starting_mol, molsPerRow=5, subImgSize=(200,200), legends=starting_mol_y)
                img.save(output_model_dir_trained + '/optimize_smiles_'+save_time+'/starting_mol.png')
        except Exception as e:
            print(e)
            pass
