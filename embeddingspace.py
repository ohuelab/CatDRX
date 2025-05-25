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
import random
# catcvae
from dataset import _dataset
from catcvae.setup import ModelArgumentParser
from catcvae.molgraph import atom_encoder_m, bond_encoder_m, atom_decoder_m, bond_decoder_m, max_atom_number, matrix_size, matrix2mol, mol2matrix
from catcvae.dataset import getDatasetFromFile, getDatasetObject, getDatasetSplittingFinetune, getDataLoader, getDataObject
from catcvae.condition import getConditionDim, getOneHotCondition
from catcvae.classweight import getClassWeight
from catcvae.loss import VAELoss, Annealer, recon_loss_fn, cosine_similarity
from catcvae.ae import CVAE, latent_space_quality, sample_latent_space
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
 
    # get datasets
    datasets_df = getDatasetFromFile(args.file, args.smiles, args.time, args.task, args.splitting, args.ids, list(args.condition_dict.keys()))
    print('datasets:', len(datasets_df))
    # get datasets object
    datasets_dobj = getDatasetObject(args, datasets_df)
    print('datasets_dobj:', len(datasets_dobj))
    # get datasets splitting
    datasets_dobj_train, datasets_dobj_val, datasets_dobj_test = getDatasetSplittingFinetune(args, datasets_df, datasets_dobj, augmentation=args.augmentation)
    print('datasets_dobj_train:', len(datasets_dobj_train))
    print('datasets_dobj_val:', len(datasets_dobj_val))
    print('datasets_dobj_test:', len(datasets_dobj_test))

    loader_train, loader_val, loader_test = getDataLoader(args, datasets_dobj_train, datasets_dobj_val, datasets_dobj_test)
    loader_all = DataLoader(datasets_dobj, batch_size=args.batch_size, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
    
    # Class weights
    if args.class_weight == 'enabled':
        if os.path.exists(args.folder_path+'/class_weights.pkl'):
            with open(args.folder_path+'/class_weights.pkl', 'rb') as f:
                class_weights = pickle.load(f)
            args.class_weights = class_weights
        else:
            class_weights = getClassWeight(datasets_dobj_train, matrix_size, args.device)
            args.class_weights = class_weights
            with open(args.folder_path+'/class_weights.pkl', 'wb') as f:
                pickle.dump(class_weights, f)
    else:
        args.class_weights = None

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
    seed_trained = args.seed
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

    save_time = time.strftime("%Y%m%d_%H%M%S")
    args.output_model_dir = output_model_dir_trained
    
    # run embedding
    mol_latent_train, mol_embedding_train, y_true_train, y_pred_train, ids_train, c_train = embed(loader_train, AE, NN_PREDICTION, device=args.device)
    # mol_latent_val, mol_embedding_val, y_true_val, y_pred_val, ids_val, c_val = embed(loader_val, AE, NN_PREDICTION, device=args.device)
    # mol_latent_test, mol_embedding_test, y_true_test, y_pred_test, ids_test, c_test = embed(loader_test, AE, NN_PREDICTION, device=args.device)
    mol_latent_all, mol_embedding_all, y_true_all, y_pred_all, ids_all, c_all = embed(loader_all, AE, NN_PREDICTION, device=args.device)

    # min max for each dimension in mol_embedding
    min_val = np.min(mol_embedding_train, axis=0)
    max_val = np.max(mol_embedding_train, axis=0)
    dimension_training = list(zip(np.floor(min_val), np.ceil(max_val)))
    print('dimension:', dimension_training)
    print('len(dimension):', len(dimension_training))

    mol_embedding_df = pd.DataFrame(mol_embedding_all.tolist())
    mol_embedding_df['y'] = y_true_all.tolist()
    # mol_embedding_df['y'] = y_pred_all.tolist()
    mol_embedding_df['ids'] = ids_all
    mol_embedding_df['c'] = ids_all
    # mol_embedding_df.reset_index(inplace=True)
    # mol_embedding_df
    
    data_subset = mol_embedding_df.drop(columns=['y', 'ids', 'c'])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_subset)
    mol_embedding_df['pca-one'] = pca_result[:,0]
    mol_embedding_df['pca-two'] = pca_result[:,1] 
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    tsne_results = tsne.fit_transform(data_subset)
    mol_embedding_df['tsne-2d-one'] = tsne_results[:,0]
    mol_embedding_df['tsne-2d-two'] = tsne_results[:,1]

    mol_embedding_df = mol_embedding_df.sort_values(by='c')

    # yield color
    hue_column = 'y'

    plt.figure(figsize=(30,12), dpi=300)

    norm = plt.Normalize(mol_embedding_df[hue_column].min(), mol_embedding_df[hue_column].max())
    sm = plt.cm.ScalarMappable(cmap="Spectral", norm=norm)
    sm.set_array([])

    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=hue_column,
        palette="Spectral",
        data=mol_embedding_df,
        legend=False,
        alpha=0.7,
        ax=ax1
    )
    ax1.figure.colorbar(sm)

    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=hue_column,
        palette="Spectral",
        data=mol_embedding_df,
        legend=None,
        alpha=0.7,
        ax=ax2
    )
    ax2.figure.colorbar(sm)

    if epoch_selected is not None:
        plt.savefig(args.output_model_dir+'/embedding_'+str(hue_column)+'_'+str(epoch_selected)+'_'+save_time+'.pdf')
    else:
        plt.savefig(args.output_model_dir+'/embedding_'+str(hue_column)+'_'+save_time+'.pdf')

    # plt.show()

    # condition color
    if "sm" in args.file:
        cat = [d['smiles_catalyst'].replace("CC(=O)O","").replace("[Pd]","").replace("...",".").replace("..",".").strip(".") for d in datasets_dobj]
        print(len(set(cat)))

        ligand_smiles = {
            'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C', 
            'P(Ph)3 ': 'c3c(P(c1ccccc1)c2ccccc2)cccc3', 
            'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C', 
            'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3', 
            'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
            'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5', 
            'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC', 
            'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]', 
            'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4', 
            'dppf': 'C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]', 
            'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C',
            'None': ''
        }

        new_ligand_smiles = dict()
        for l in ligand_smiles:
            new_ligand_smiles[Chem.MolToSmiles(Chem.MolFromSmiles(ligand_smiles[l]))] = l

        new_cat = list()
        new_reactant1 = list()
        new_reactant2 = list()
        new_reagent = list()
        new_solvent = list()
        read_file = pd.read_csv('dataset/sm_test1.csv')
        for i in mol_embedding_df['ids']:
            reactant1 = read_file[read_file['index'] == int(i)]['Reactant_1_Name'].values[0]
            reactant2 = read_file[read_file['index'] == int(i)]['Reactant_2_Name'].values[0]
            reagent = read_file[read_file['index'] == int(i)]['Reagent_1_Short_Hand'].values[0]
            solvent = read_file[read_file['index'] == int(i)]['Solvent_1_Short_Hand'].values[0]
            ligand = read_file[read_file['index'] == int(i)]['Ligand_Short_Hand'].values[0]
            new_reactant1.append(reactant1)
            new_reactant2.append(reactant2)
            new_reagent.append(reagent)
            new_solvent.append(solvent)
            if str(ligand) != '' and str(ligand) != 'None' and str(ligand) != 'nan':
                new_cat.append(ligand)
            else:
                new_cat.append('None')
       
        print("catalyst", len(new_cat))
        print("reactant1", len(new_reactant1))
        print("reactant2", len(new_reactant2))
        print("reagent", len(new_reagent))
        print("solvent", len(new_solvent))

        mol_embedding_df['cat'] = new_cat
        mol_embedding_df['reactant1'] = new_reactant1
        mol_embedding_df['reactant2'] = new_reactant2
        mol_embedding_df['reagent'] = new_reagent
        mol_embedding_df['solvent'] = new_solvent

        for component in ['cat', 'reactant1', 'reactant2', 'reagent', 'solvent']:
            hue_column = component
            palette = sns.color_palette("husl", len(mol_embedding_df[hue_column].unique()))

            plt.figure(figsize=(30,13), dpi=300)

            ax1 = plt.subplot(1, 2, 1)
            sns.scatterplot(
                x="pca-one", y="pca-two",
                hue=hue_column,
                palette=palette,
                data=mol_embedding_df,
                legend=True,
                alpha=0.7,
                ax=ax1
            )
            ax1.legend(loc='upper right')

            ax2 = plt.subplot(1, 2, 2)
            sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue=hue_column,
                palette=palette,
                data=mol_embedding_df,
                legend=True,
                alpha=0.7,
                ax=ax2
            )
            # ax2.legend(loc='upper right')

            if epoch_selected is not None:
                plt.savefig(args.output_model_dir+'/embedding_'+str(hue_column)+'_'+str(epoch_selected)+'_'+save_time+'.pdf')
            else:
                plt.savefig(args.output_model_dir+'/embedding_'+str(hue_column)+'_'+save_time+'.pdf')

            # plt.show()

    if "bh" in args.file:
        new_reactant1 = list()
        new_reagent1 = list()
        new_reagent2 = list()
        new_cat = list()
        read_file = pd.read_csv('dataset/bh_test1.csv')
        for i in mol_embedding_df['ids']:
            reactant1 = read_file[read_file['index'] == int(i)]['Aryl halide'].values[0]
            reagent1 = read_file[read_file['index'] == int(i)]['Base'].values[0]
            reagent2 = read_file[read_file['index'] == int(i)]['Additive'].values[0]
            ligand = read_file[read_file['index'] == int(i)]['Ligand'].values[0]
            new_reactant1.append(reactant1)
            new_reagent1.append(reagent1)
            new_reagent2.append(reagent2)
            if str(ligand) != '' and str(ligand) != 'None' and str(ligand) != 'nan':
                new_cat.append(ligand)
            else:
                new_cat.append('None')
       
        print("catalyst", len(new_cat))
        print("reactant1", len(new_reactant1))
        print("reagent1", len(new_reagent1))
        print("reagent2", len(new_reagent2))

        mol_embedding_df['cat'] = new_cat
        mol_embedding_df['reactant1'] = new_reactant1
        mol_embedding_df['reagent1'] = new_reagent1
        mol_embedding_df['reagent2'] = new_reagent2

        for component in ['cat', 'reactant1', 'reagent1', 'reagent2']:
            hue_column = component
            palette = sns.color_palette("husl", len(mol_embedding_df[hue_column].unique()))

            plt.figure(figsize=(30,13), dpi=300)

            ax1 = plt.subplot(1, 2, 1)
            sns.scatterplot(
                x="pca-one", y="pca-two",
                hue=hue_column,
                palette=palette,
                data=mol_embedding_df,
                legend=True,
                alpha=0.7,
                ax=ax1
            )
            ax1.legend(loc='upper right')

            ax2 = plt.subplot(1, 2, 2)
            sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue=hue_column,
                palette=palette,
                data=mol_embedding_df,
                legend=True,
                alpha=0.7,
                ax=ax2
            )
            ax2.legend(loc='upper right')

            if epoch_selected is not None:
                plt.savefig(args.output_model_dir+'/embedding_'+str(hue_column)+'_'+str(epoch_selected)+'_'+save_time+'.pdf')
            else:
                plt.savefig(args.output_model_dir+'/embedding_'+str(hue_column)+'_'+save_time+'.pdf')

            # plt.show()

    # Random molecules and conditions
    from catcvae.condition import getSampleCondition
    from rdkit.Chem import Descriptors
    import random

    sample_condition = 5
    sample_molecule = 200
    cond_random = list()

    mt_15 = list()
    # mt_15 = [2, 10, 51, 182, 393, 17, 71, 90, 150]
    # mt_15 = [1,2,3,4,5]
    # select specific dataset based on y value
    # for dd in range(len(datasets_dobj_train)):
    #     print(dd, datasets_dobj_train[dd]['y'].item())
    #     if datasets_dobj_train[dd]['y'].item() >= 1.5:
    #         mt_15.append(dd)

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
    
    # check condition statistics using reactant, reagent, and time
    def dataset_condition_statisitic(c_detail):
        dataset_condition = list()
        for d in datasets_dobj:
            if d['smiles_reactant'] == c_detail['smiles_reactant'] and \
               d['smiles_reagent'] == c_detail['smiles_reagent'] and \
               d['time_h'] == c_detail['time_h']:
                dataset_condition.append(d)
        return dataset_condition

    # get ECFP fingerprint
    def getECFP(smiles, radius=2, nBits=2048, useChirality=False):
        try:
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
            return list(fp)
        except Exception as e:
            print('getECFP ERROR SMILES:', smiles)
            return None
        
    latent_random = list()
    latent_random_cond = list()
    latent_random_c = list()
    latent_random_y = list()
    sample_mol = list()
    sample_mol_smiles = list()
    sample_mol_fp = list()
    
    AE.eval()
    NN_PREDICTION.eval()
    with torch.no_grad():
        for i in tqdm(range(sample_condition)):
        # for i in tqdm(mt_15):
            sample_condition_latent_random = list()
            sample_condition_latent_random_c = list()
            sample_condition_latent_random_y = list()
            sample_condition_sample_mol_fp = list()
            sample_condition_sample_mol_smiles = list()

            # _mol_latent, cond_random, c_detail = random_dataset(index=i)
            _mol_latent, cond_random, c_detail = random_dataset()
            dataset_condition = dataset_condition_statisitic(c_detail)
            for j in range(sample_molecule):
                AE.eval()
                NN_PREDICTION.eval()
                c = torch.tensor(np.array(cond_random), dtype=torch.float).type(torch.FloatTensor).to(args.device)
                # out_matrix, out_decoded, latent_point = sample_latent_space(AE, c=c, device=args.device)
                out_matrix, out_decoded, latent_point = sample_latent_space(AE, c=c, guide_tensor=None, dim_range=dimension_training, device=args.device)
                
                try:
                    mol_catalyst = matrix2mol(out_matrix, correct=True)

                    AE.eval()
                    NN_PREDICTION.eval()
                    mol_catalyst.UpdatePropertyCache(strict=False)
                    smiles_catalyst = Chem.MolToSmiles(mol_catalyst)
                    mol_recheck = Chem.MolFromSmiles(smiles_catalyst)
                    try:
                        smiles_catalyst = Chem.CanonSmiles(smiles_catalyst, useChiral=False)
                    except Exception as e:
                        print('SMILES:', smiles_catalyst)
                        print('(CanonSmiles) ERROR!:', e)
                        continue
                        # pass
                    ecfp = getECFP(smiles_catalyst, radius=2, nBits=2048, useChirality=False)
                    assert ecfp is not None

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

                    # latent_random.append(latent_point.squeeze(0).detach().cpu().numpy())
                    # latent_random_cond.append(np.concatenate((latent_point.squeeze(0).detach().cpu().numpy(), c.squeeze(0).detach().cpu().numpy())))
                    latent_random.append(mol_embedding_t.squeeze(0))
                    latent_random_y.append(sample_y_pred_t)
                    latent_random_c.append('Condition '+str(i))
                    sample_mol_fp.append(ecfp)
                    sample_mol_smiles.append(smiles_catalyst)

                    # sample_condition_latent_random.append(latent_point.squeeze(0).detach().cpu().numpy())
                    # latent_random_cond.append(np.concatenate((latent_point.squeeze(0).detach().cpu().numpy(), c.squeeze(0).detach().cpu().numpy())))
                    sample_condition_latent_random.append(mol_embedding_t.squeeze(0))
                    sample_condition_latent_random_y.append(sample_y_pred_t)
                    sample_condition_latent_random_c.append('Condition '+str(i+1))
                    sample_condition_sample_mol_fp.append(ecfp)
                    sample_condition_sample_mol_smiles.append(smiles_catalyst)

                except Exception as e:
                    print('(ProcessMol) ERROR!', e)
                    # save error out_matrix as numpy object
                    # np.savez(output_model_dir_trained+'/error_mol_'+str(i)+'_'+str(j)+'.npz', out_matrix=out_matrix.cpu())

            y_sample = sample_condition_latent_random_y
            y_condition = [d['y'].item() for d in dataset_condition]

            if len(y_condition) > 10:
                sns.set_style(style='ticks')
                # plot distribution
                color_1 = '#1C3077' # blue
                color_2 = '#E97132' # orange
                plt.figure(figsize=(8, 3), dpi=100)
                # sns.histplot(y_condition, bins=np.arange(-0.2, 4.2, 0.1), color=color_2, stat='percent')
                # sns.histplot(y_sample, bins=np.arange(-0.2, 4.2, 0.1), color=color_1, stat='percent')
                sns.kdeplot(y_condition, color=color_2, bw_adjust=0.5, fill=True, alpha=0.25)
                sns.kdeplot(y_sample, color=color_1, bw_adjust=0.5, fill=True, alpha=0.25)
                # plt.xlabel('Predicted and dataset value')
                # plt.ylabel('Percentage of molecules (%)')
                # plt.xlim(-0.2, 4.2)
                plt.xlim(-0.5,2.5)
                plt.title('Condition'+str(i+1))
                plt.legend(['Dataset catalyst', 'Generated catalyst'])
                sns.despine()
                # plt.show()
                # plt.savefig(args.output_model_dir+'/distribution_'+str(i+1)+'_'+save_time+'.pdf')

            # save smiles and predicted y list
            # with open(args.output_model_dir+'/distribution_sample_condition_'+save_time+'.txt', 'a') as f:
            #     for k in range(len(sample_condition_latent_random_y)):
            #         f.write(str(i+1) + ', ' + str(sample_condition_sample_mol_smiles[k]) + ', ' + str(sample_condition_latent_random_y[k]) + '\n')

    mol_embedding_df = pd.DataFrame(latent_random)
    # mol_embedding_df = pd.DataFrame(latent_random_cond)
    mol_embedding_df['y'] = latent_random_y
    mol_embedding_df['c'] = latent_random_c
    data_subset = mol_embedding_df.drop(columns=['y', 'c'])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_subset)
    mol_embedding_df['pca-one'] = pca_result[:,0]
    mol_embedding_df['pca-two'] = pca_result[:,1] 
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    # tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=500)
    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    tsne_results = tsne.fit_transform(data_subset)
    mol_embedding_df['tsne-2d-one'] = tsne_results[:,0]
    mol_embedding_df['tsne-2d-two'] = tsne_results[:,1]

    # yield color
    hue_column = 'y'

    plt.figure(figsize=(30,12), dpi=300)

    # norm = plt.Normalize(mol_embedding_df[hue_column].min(), mol_embedding_df[hue_column].max())
    norm = plt.Normalize(0,100)
    sm = plt.cm.ScalarMappable(cmap="Spectral", norm=norm)
    sm.set_array([])

    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=hue_column,
        palette="Spectral",
        data=mol_embedding_df,
        legend=False,
        alpha=0.7,
        ax=ax1
    )
    ax1.figure.colorbar(sm)

    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=hue_column,
        palette="Spectral",
        data=mol_embedding_df,
        legend=None,
        alpha=0.7,
        ax=ax2
    )
    ax2.figure.colorbar(sm)

    if epoch_selected is not None:
        plt.savefig(args.output_model_dir+'/embedding_random_'+str(hue_column)+'_'+str(epoch_selected)+'_'+save_time+'.pdf')
    else:
        plt.savefig(args.output_model_dir+'/embedding_random_'+str(hue_column)+'_'+save_time+'.pdf')

    # plt.show()

    # condition color
    hue_column = 'c'

    plt.figure(figsize=(30,13), dpi=300)

    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=hue_column,
        palette="Dark2",
        data=mol_embedding_df,
        legend=True,
        alpha=0.7,
        ax=ax1
    )
    ax1.legend(loc='upper right')

    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=hue_column,
        palette="Dark2",
        data=mol_embedding_df,
        legend=True,
        alpha=0.7,
        ax=ax2
    )
    ax2.legend(loc='upper right')

    if epoch_selected is not None:
        plt.savefig(args.output_model_dir+'/embedding_random_'+str(hue_column)+'_'+str(epoch_selected)+'_'+save_time+'.pdf')
    else:
        plt.savefig(args.output_model_dir+'/embedding_random_'+str(hue_column)+'_'+save_time+'.pdf')

    # plt.show()

    mol_embedding_df = pd.DataFrame(sample_mol_fp)
    # mol_embedding_df = pd.DataFrame(latent_random_cond)
    mol_embedding_df['y'] = latent_random_y
    mol_embedding_df['c'] = latent_random_c
    data_subset = mol_embedding_df.drop(columns=['y', 'c'])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_subset)
    mol_embedding_df['pca-one'] = pca_result[:,0]
    mol_embedding_df['pca-two'] = pca_result[:,1] 
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    # tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    tsne_results = tsne.fit_transform(data_subset)
    mol_embedding_df['tsne-2d-one'] = tsne_results[:,0]
    mol_embedding_df['tsne-2d-two'] = tsne_results[:,1]

    # yield color
    hue_column = 'y'
    mark_column = 'c'

    plt.figure(figsize=(30,12), dpi=300)

    # norm = plt.Normalize(mol_embedding_df[hue_column].min(), mol_embedding_df[hue_column].max())
    norm = plt.Normalize(0, 100)
    sm = plt.cm.ScalarMappable(cmap="Spectral", norm=norm)
    sm.set_array([])

    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=hue_column,
        palette="Spectral",
        data=mol_embedding_df,
        legend=False,
        alpha=0.7,
        style=mark_column,
        s=100,
        ax=ax1,
    )
    ax1.figure.colorbar(sm)

    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=hue_column,
        palette="Spectral",
        data=mol_embedding_df,
        legend=True,
        alpha=0.7,
        style=mark_column,
        s=100,
        ax=ax2
    )
    
    ax2.figure.colorbar(sm)

    # produce a legend with a cross-section of sizes from the scatter
    handles, labels = ax2.get_legend_handles_labels()
    handles_c, labels_c = zip(*[(h, l) for h, l in zip(handles, labels) if 'C' in l])
    legend2 = ax2.legend(handles_c, labels_c, loc="upper right", title="Condition")
    

    if epoch_selected is not None:
        plt.savefig(args.output_model_dir+'/embedding_random_fps_'+str(hue_column)+'_'+str(epoch_selected)+'_'+save_time+'.pdf')
    else:
        plt.savefig(args.output_model_dir+'/embedding_random_fps_'+str(hue_column)+'_'+save_time+'.pdf')

    # plt.show()
