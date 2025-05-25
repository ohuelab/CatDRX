# Library
import os
from rdkit import Chem
import time
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
import torch.optim as optim
import torch
from torch import nn
import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from torch_geometric.loader import DataLoader
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sns.set_theme(style="white", palette=None)
# catcvae
from dataset import _dataset
from catcvae.setup import ModelArgumentParser
from catcvae.molgraph import atom_encoder_m, bond_encoder_m, atom_decoder_m, bond_decoder_m, max_atom_number, matrix_size, matrix2mol, mol2matrix
from catcvae.dataset import getDatasetFromFile, getDatasetObject, getDatasetSplittingFinetune, getDataLoader, getDataObject
from catcvae.condition import getConditionDim, getOneHotCondition
from catcvae.classweight import getClassWeight
from catcvae.loss import VAELoss, Annealer, recon_loss_fn, cosine_similarity
from catcvae.ae import CVAE
from catcvae.ae import latent_space_quality, sample_latent_space
from catcvae.training import save_model, save_loss, save_report, save_model_latest
from catcvae.prediction import NN, NN_TASK
from catcvae.latent import embed, save_latent
from catcvae.condition import getSampleCondition
from catcvae.metrics import *

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Main
if __name__ == '__main__':
    
    # argument setup
    parser = ModelArgumentParser()
    args = parser.setArgument()
    
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
    print(AE)
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
    if epoch_selected is not None:
        AE.load_state_dict(torch.load(output_model_dir_trained + '/model_ae_'+str(epoch_selected)+'.pth', map_location=torch.device(args.device)))
        NN_PREDICTION.load_state_dict(torch.load(output_model_dir_trained + '/model_nn_'+str(epoch_selected)+'.pth', map_location=torch.device(args.device)))
    else:    
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

    from catcvae.condition import getSampleCondition
    from rdkit.Chem import Descriptors
    import random

    # random generation

    # select from training dataset (random or index)
    def random_dataset(index=None):
        AE.eval()
        NN_PREDICTION.eval()
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
    
    def define_guide(smiles):
        guide_smiles = smiles
        guide_mol = smiles_to_mol(guide_smiles)
        guide_matrix = mol2matrix(guide_mol)
        guide_matrix = torch.tensor(guide_matrix, dtype=torch.float).type(torch.FloatTensor).to(args.device)
        return guide_matrix.view(1, -1)

    with torch.no_grad():
        sample_molecules_reached = 0
        sample_molecules_all = 0
        sample_molecules = 1000
        sample_mol_dict = dict()
        sample_mol = []
        sample_smiles = []
        sample_mol_text = []
        invalid_mol = 0
        correction = args.correction == 'enabled'
        from_around_mol = args.from_around_mol == 'enabled'
        from_mol_condition = args.from_around_mol_cond == 'enabled'
        from_training_space = args.from_training_space == 'enabled'
        from_guide = args.from_guide != str(None)
        

        while sample_molecules_all < sample_molecules:
            AE.eval()
            NN_PREDICTION.eval()
            print("Running:", sample_molecules_reached, "/", sample_molecules_all, "/", sample_molecules)
            sample_molecules_all += 1

            mol_latent, cond_random, c_detail = random_dataset()

            guide_tensor = define_guide(str(args.from_guide)) if from_guide else None
            print('Guide:', str(args.from_guide))

            if from_around_mol:
                if from_mol_condition:
                    c = torch.tensor(np.array(cond_random), dtype=torch.float).type(torch.FloatTensor).to(args.device)
                    out_matrix, out_decoded, latent_point = sample_latent_space(AE, c=c, mol_latent=mol_latent, guide_tensor=guide_tensor, noise=True, device=args.device)
                elif not from_mol_condition:
                    mol_latent_other, cond_random_other, c_detail_other = random_dataset()
                    c_detail = c_detail_other
                    c_other = torch.tensor(np.array(cond_random_other), dtype=torch.float).type(torch.FloatTensor).to(args.device)
                    out_matrix, out_decoded, latent_point = sample_latent_space(AE, c=c_other, mol_latent=mol_latent, guide_tensor=guide_tensor, noise=True, device=args.device)
            elif not from_around_mol:
                c = torch.tensor(np.array(cond_random), dtype=torch.float).type(torch.FloatTensor).to(args.device)
                if from_training_space:
                    out_matrix, out_decoded, latent_point = sample_latent_space(AE, c=c, guide_tensor=guide_tensor, dim_range=dimension_training, device=args.device)
                else:
                    out_matrix, out_decoded, latent_point = sample_latent_space(AE, c=c, guide_tensor=guide_tensor, device=args.device)

            try:
                mol_catalyst = matrix2mol(out_matrix, correct=correction)

                AE.eval()
                NN_PREDICTION.eval()
                mol_catalyst.UpdatePropertyCache(strict=False)
                smiles_catalyst = Chem.MolToSmiles(mol_catalyst)
                mol_recheck = Chem.MolFromSmiles(smiles_catalyst)
                try:
                    smiles_catalyst = Chem.CanonSmiles(smiles_catalyst, useChiral=False)
                except Exception as e:
                    print('SMILES:', smiles_catalyst)
                    print('(CanonSmiles) ERROR!', e)
                    invalid_mol += 1
                    continue
                    # pass

                sample_mol.append(mol_catalyst)

                d = {
                    "X_reactant": c_detail['smiles_reactant'],
                    "X_reagent": c_detail['smiles_reagent'],
                    "X_product": c_detail['smiles_product'],
                    "X_catalyst": smiles_catalyst,
                    "X_time": c_detail['time_h'],
                    "y": 0.0, # dummy
                    "ids": 0, # dummy
                }
                d['C_'+list(args.condition_dict.keys())[0]] = Descriptors.ExactMolWt(mol_catalyst) # should be dummy?
                # d['C_'+list(args.condition_dict.keys())[0]] = 0 # dummy
                
                    
                sample_mol_t = getDataObject(args, d)
                loader_mol_t = DataLoader([sample_mol_t], batch_size=1, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
                mol_latent_t, mol_embedding_t, y_true_t, y_pred_t, ids_t, c_t = embed(loader_mol_t, AE, NN_PREDICTION, device=args.device)
                sample_y_pred_t = y_pred_t.item()

                sample_mol_text.append(str(sample_y_pred_t))
                sample_mol_dict.setdefault(smiles_catalyst, []).append(sample_y_pred_t)
                sample_smiles.append((smiles_catalyst, sample_y_pred_t))
                sample_molecules_reached += 1

            except Exception as e:
                print('(ProcessMol) ERROR!', e)
                # save error out_matrix as numpy object
                # np.savez(output_model_dir_trained+'/error_mol_'+str(sample_molecules_all)+'.npz', out_matrix=out_matrix.cpu())
                invalid_mol += 1

    m_validity = sample_molecules-invalid_mol
    m_validity_percent = m_validity/sample_molecules*100 if sample_molecules != 0 else 0

    m_uniqueness = len(set(sample_mol_dict.keys()))
    m_uniqueness_percent = m_uniqueness/m_validity*100 if m_validity != 0 else 0

    training_smiles = [Chem.CanonSmiles(t.smiles_catalyst, useChiral=False) for t in datasets_dobj_train]
    m_novelty = len(set(sample_mol_dict.keys()) - set(training_smiles))
    m_novelty_percent = m_novelty/m_uniqueness*100 if m_uniqueness != 0 else 0

    print('valid_1: {} {:.2f}%'.format(m_validity, m_validity_percent))
    print('unique_1: {} {:.2f}%'.format(m_uniqueness, m_uniqueness_percent))
    print('novel_1: {} {:.2f}%'.format(m_novelty, m_novelty_percent))

    smiles_list = [s[0] for s in sample_smiles]
    # smiles_list = normalize(smiles_list)
    # training_smiles = normalize(training_smiles)

    if 'sm' in args.file:
        m_validity_sm = 0
        for s in smiles_list:
            frag1 = 'CC(=O)O'
            frag2 = 'CC(=O)O'
            frag3 = '[Pd]'
            frags = s.split(".")
            if len(frags) == 4:
                if frag1 in frags and frag2 in frags and frag3 in frags and frags.count(frag1) == 2:
                    m_validity_sm += 1
            elif len(frags) == 3:
                if frag1 in frags and frag2 in frags and frag3 in frags and frags.count(frag1) == 2:
                    m_validity_sm += 1
        m_validity_sm_percent = m_validity_sm/sample_molecules*100 if sample_molecules != 0 else 0
        print('valid_sm: {} {:.2f}%'.format(m_validity_sm, m_validity_sm_percent))

    if 'cc' in args.file:
        metal = ['Pd', 'Ni', 'Cu', 'Ag', 'Au', 'Pt']
        m_validity_3frag = 0
        for s in smiles_list:
            mol_frag = Chem.MolFromSmiles(s)
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
                    if flag_metal:
                        m_validity_3frag += 1
            except Exception as e:
                pass
        m_validity_3frag_percent = m_validity_3frag/sample_molecules*100 if sample_molecules != 0 else 0
        print('valid_3frag: {} {:.2f}%'.format(m_validity_3frag, m_validity_3frag_percent))

    if 'ps' in args.file:
        m_validity_1frag = 0
        for s in smiles_list:
            mol_frag = Chem.MolFromSmiles(s)
            try:
                rs = Chem.GetMolFrags(mol_frag, asMols=True)
                if len(rs) == 1:
                    m_validity_1frag += 1
            except Exception as e:
                pass
        m_validity_1frag_percent = m_validity_1frag/sample_molecules*100 if sample_molecules != 0 else 0
        print('valid_1frag: {} {:.2f}%'.format(m_validity_1frag, m_validity_1frag_percent))

    # calculate metrics
    m_validity_2 = validity(smiles_list, num_samples=sample_molecules)
    m_uniqueness_2 = uniqueness(smiles_list)
    m_novelty_2 = novelty(set(smiles_list), training_smiles)
    print('valid_2:', m_validity_2)
    print('unique_2:', m_uniqueness_2)
    print('novel_2:', m_novelty_2)

    dictionary = get_fingerprint_dictionary(smiles_list+training_smiles)
    m_internal_diversity = internal_diversity(smiles_list, dictionary=dictionary)
    m_similarity_to_nearest_neighbor = similarity_to_nearest_neighbor(smiles_list, training_smiles, dictionary=dictionary)
    m_frechet_distance = frechet_distance(smiles_list, training_smiles)
    print('InDiv:', m_internal_diversity)
    print('SNN:', m_similarity_to_nearest_neighbor)
    print('FCD:', m_frechet_distance)

    # save results
    save_time = time.strftime("%Y%m%d_%H%M%S")
    scheme = ''
    if from_training_space: scheme+='trainspace_'
    if from_around_mol: scheme+='lat_'
    if from_mol_condition: scheme+='con_'
    if not correction: scheme+='nocorrect_'
    
    with open(output_model_dir_trained+'/generated_stats_'+scheme+save_time+'.txt', 'w') as f:
        f.write('Validity,{},,{:.4f}\n'.format(m_validity, m_validity_percent))
        if 'aap9112' in args.file:
            f.write('Validity (Task),{},,{:.4f}\n'.format(m_validity_sm, m_validity_sm_percent))
        if 'suzuki' in args.file:
            f.write('Validity (Task),{},,{:.4f}\n'.format(m_validity_3frag, m_validity_3frag_percent))
        if 'ps_ddG' in args.file:
            f.write('Validity (Task),{},,{:.4f}\n'.format(m_validity_1frag, m_validity_1frag_percent))
        f.write('Uniqueness,{},,{:.4f}\n'.format(m_uniqueness, m_uniqueness_percent))
        f.write('Novelty,{},,{:.4f}\n'.format(m_novelty, m_novelty_percent))
        f.write('IntDiv,,,{},{}\n'.format(m_internal_diversity[0], m_internal_diversity[1]))
        f.write('SNN,,,{},{}\n'.format(m_similarity_to_nearest_neighbor[0], m_similarity_to_nearest_neighbor[1]))
        f.write('FCD,,,{}\n'.format(m_frechet_distance))
        f.write('\n')
        f.write('Confirmation:\n')
        f.write('Validity_2,,,{}\n'.format(m_validity_2))
        f.write('Uniqueness_2,,,{}\n'.format(m_uniqueness_2))
        f.write('Novelty_2,,,{}\n'.format(m_novelty_2))

    # save to text file
    with open(output_model_dir_trained+'/generated_mol_'+scheme+save_time+'.csv', 'w') as f:
        for s in sample_smiles:
            f.write(str(s[0])+','+str(s[1])+'\n')

    
    
