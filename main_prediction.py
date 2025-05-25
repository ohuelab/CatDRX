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
from catcvae.ae import CVAE, latent_space_quality, sample_latent_space
from catcvae.training import save_model, save_loss, save_report, save_model_latest, save_model_latest_temp, write_continue_training
from catcvae.prediction import NN
from catcvae.latent import embed, save_latent

# Main
if __name__ == '__main__':
    start_time_run = time.time()
    
    # argument setup
    parser = ModelArgumentParser()
    args = parser.setArgument()

    # train dataset
    path_train = args.folder_path+'/datasets_dobj_train_'+str(args.seed)+'.pkl'
    if os.path.exists(path_train):
        # get datasets splitting
        datasets_dobj_train, datasets_dobj_val, datasets_dobj_test = getDatasetSplitting(args)
        print('datasets_dobj_train:', len(datasets_dobj_train))
        print('datasets_dobj_val:', len(datasets_dobj_val))
        print('datasets_dobj_test:', len(datasets_dobj_test))
    else:
        # get datasets
        datasets_df = getDatasetFromFile(args.file, args.smiles, args.time, args.task, args.splitting, args.ids, list(args.condition_dict.keys()))
        print('datasets:', len(datasets_df))
        # get datasets object
        datasets_dobj = getDatasetObject(args, datasets_df)
        print('datasets_dobj:', len(datasets_dobj))
        # get datasets splitting
        datasets_dobj_train, datasets_dobj_val, datasets_dobj_test = getDatasetSplitting(args, datasets_df, datasets_dobj, augmentation=args.augmentation)
        print('datasets_dobj_train:', len(datasets_dobj_train))
        print('datasets_dobj_val:', len(datasets_dobj_val))
        print('datasets_dobj_test:', len(datasets_dobj_test))

    loader_train, loader_val, loader_test = getDataLoader(args, datasets_dobj_train, datasets_dobj_val, datasets_dobj_test)
    # loader_all = DataLoader(datasets_dobj, batch_size=args.batch_size, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])

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
    NN_PREDICTION = NN(in_dim=args.emb_dim+(3*args.emb_cond_dim)+args.cond_dim, out_dim_class=1).to(args.device)
    print(NN_PREDICTION)
    # setup loss function
    if args.annealing:
        annealing_agent = Annealer(total_steps=args.annealing_slope_length, shape=args.annealing_shape)
    else:
        annealing_agent = None
    LOSSCLASS = VAELoss(args.AE_loss, class_weights=args.class_weights, annealer=annealing_agent, device=args.device)
    print(LOSSCLASS)
    loss_fn_nn = nn.L1Loss()
    print(loss_fn_nn)

    # setup parameters
    model_param_group = [{'params': AE.parameters(), 'lr': args.lr},
                        {'params': NN_PREDICTION.parameters(), 'lr': args.lr}]
    # setup optimizers
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = np.inf

    # setup output directory
    args.time = time.strftime("%Y%m%d_%H%M%S")
    args.output_model_dir = 'dataset/'+args.file+'/output'+'_'+str(args.seed)+'_'+str(args.time)
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
        with open(args.output_model_dir + '/setup.txt', "a") as f:
            for a in vars(args):
                f.write(a + ' ' + str(vars(args)[a]) + '\n')
    # write continue training file
    write_continue_training(args)
    
    # training function
    def train(args, loader, optimizer):
        AE.train()
        NN_PREDICTION.train()

        AE_loss_accum = 0
        recon_loss_accum = 0
        kl_loss_accum = 0
        nn_loss_accum = 0

        for batch in loader:
            batch = batch.to(args.device)

            (x_target, y_decoded, mu, varlog, z, c) = AE(batch)

            # print('x_target:', x_target.size())
            # print('y_decoded:', y_decoded.size())
            # print('mu:', mu.size())
            # print('varlog:', varlog.size())
            # print('z:', z.size())
            # print('c:', c.size())
            AE_loss, all_loss = LOSSCLASS(y_decoded, x_target, mu, varlog, alpha=args.alpha, beta=args.beta, device=args.device)

            y_pred = NN_PREDICTION(mu, c)
            nn_loss = loss_fn_nn(torch.squeeze(y_pred), batch.y)
            all_loss['nn_loss'] = nn_loss

            loss = AE_loss+nn_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            AE_loss_accum += AE_loss.detach().cpu().item()+nn_loss.detach().cpu().item()
            recon_loss_accum += all_loss['recon_loss']
            kl_loss_accum += all_loss['kl_loss']
            nn_loss_accum += all_loss['nn_loss']

        AE_loss_accum /= len(loader)
        recon_loss_accum /= len(loader)
        kl_loss_accum /= len(loader)
        nn_loss_accum /= len(loader)

        return AE_loss_accum, {'recon_loss': recon_loss_accum, 'kl_loss': kl_loss_accum, 'nn_loss': nn_loss_accum}

    # validation function
    def validate(args, loader):
        AE.eval()
        NN_PREDICTION.eval()

        with torch.no_grad():
            AE_loss_accum = 0
            recon_loss_accum = 0
            kl_loss_accum = 0
            nn_loss_accum = 0

            for batch in loader:
                batch = batch.to(args.device)

                (x_target, y_decoded, mu, varlog, z, c) = AE(batch)
                
                # print('x_target:', x_target.size())
                # print('y_decoded:', y_decoded.size())
                # print('mu:', mu.size())
                # print('varlog:', varlog.size())
                # print('z:', z.size())
                # print('c:', c.size())
                AE_loss, all_loss = LOSSCLASS(y_decoded, x_target, mu, varlog, alpha=args.alpha, beta=args.beta, device=args.device)

                y_pred = NN_PREDICTION(mu, c)
                nn_loss = loss_fn_nn(torch.squeeze(y_pred), batch.y)
                all_loss['nn_loss'] = nn_loss

                AE_loss_accum += AE_loss.detach().cpu().item()+nn_loss.detach().cpu().item()
                recon_loss_accum += all_loss['recon_loss']
                kl_loss_accum += all_loss['kl_loss']
                nn_loss_accum += all_loss['nn_loss']
            
            AE_loss_accum /= len(loader)
            recon_loss_accum /= len(loader)
            kl_loss_accum /= len(loader)
            nn_loss_accum /= len(loader)

            return AE_loss_accum, {'recon_loss': recon_loss_accum, 'kl_loss': kl_loss_accum, 'nn_loss': nn_loss_accum}

    # training model function
    def train_model(args, loader_train, loader_val, optimizer, epoch):
        start_time = time.time()

        AE_loss_accum_train, all_loss_train = train(args, loader_train, optimizer)
        AE_loss_accum_val, all_loss_val = validate(args, loader_val)

        global optimal_loss
        temp_loss = AE_loss_accum_val
        if temp_loss < optimal_loss:
            optimal_loss = temp_loss
            save_model(args.output_model_dir, optimal_loss, AE, nn=NN_PREDICTION)
            # save_latent(args.output_model_dir, loader_all, AE, NN_PREDICTION)

        # if epoch%500 == 0 or epoch == args.epochs-1:
        #     save_model(args.output_model_dir, temp_loss, AE, nn=NN_PREDICTION, save_best=False, epoch=epoch)
        #     save_latent(args.output_model_dir, loader_all, AE, NN_PREDICTION, epoch, device=args.device)
        # if epoch%50 == 0:
        #     save_model_latest_temp(args.output_model_dir, optimizer, LOSSCLASS, optimal_loss, AE, nn=NN_PREDICTION, epoch=epoch)
        # save_model_latest(args.output_model_dir, optimizer, LOSSCLASS, optimal_loss, AE, nn=NN_PREDICTION, epoch=epoch)

        sample_num = 10
        corr, unique, sample_smiles = latent_space_quality(AE, ae_type=args.AE_type, sample_num=sample_num, datasets_dobj=datasets_dobj_train, device=args.device)
        validity = corr * 100. / sample_num
        diversity = unique * 100. / sample_num

        # save to text
        with open(args.output_model_dir + '/sample_smiles.txt', "w") as f:
            f.write('epoch: '+str(epoch)+' ')
            f.write('validity: '+str(validity)+' ')
            f.write('diversity: '+str(diversity)+' ')
            for s in sample_smiles:
                f.write(s+'\n')
        # save to text
        with open(args.output_model_dir + '/sample_smiles_all.txt', "a") as f:
            f.write('epoch: '+str(epoch)+' ')
            f.write('validity: '+str(validity)+' ')
            f.write('diversity: '+str(diversity)+'\n')
            for s in sample_smiles:
                f.write(s+'\n')
                break
        if epoch%100 == 0 or epoch == args.epochs-1:
            try:
                from rdkit.Chem import Draw
                if len(sample_smiles) > 0:
                    mols = list()
                    for s in sample_smiles:
                        try: 
                            mol = Chem.MolFromSmiles(s)
                            if mol:
                                mols.append(mol)
                            else:
                                mol = Chem.MolFromSmiles(s, sanitize=False)
                                if mol:
                                    mols.append(mol)
                        except:
                            pass
                    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200,200))
                    img.save(args.output_model_dir + '/sample_smiles.png')
            except:
                pass
            

        print('AE Loss Train: {:.5f}\tAE Loss Val: {:.5f}\tValidity: {:.5f}\tDiversity: {:.5f}\tTime: {:.5f}'.
            format(AE_loss_accum_train, AE_loss_accum_val, validity, diversity, time.time() - start_time))

        save_report(args.output_model_dir, epoch, AE_loss_accum_train, AE_loss_accum_val, optimal_loss, validity, diversity)
        save_loss(args.output_model_dir, epoch, all_loss_train, all_loss_val, LOSSCLASS.annealing_agent.current_step, LOSSCLASS.annealing_agent.slope())

        LOSSCLASS.annealing_agent.step()

        return 

    # start training
    for epoch in tqdm(range(args.epochs)):
        print('epoch: {}'.format(epoch))
        train_model(args, loader_train, loader_val, optimizer, epoch)
        # limit only time 22 hours
        if time.time() - start_time_run > 22*60*60:
            save_model_latest(args.output_model_dir, optimizer, LOSSCLASS, optimal_loss, AE, nn=NN_PREDICTION, epoch=epoch)
            break

    # load best model
    epoch_selected = None
    if epoch_selected is not None:
        AE.load_state_dict(torch.load(args.output_model_dir + '/model_ae_'+str(epoch_selected)+'.pth', map_location=torch.device(args.device)))
        NN_PREDICTION.load_state_dict(torch.load(args.output_model_dir + '/model_nn_'+str(epoch_selected)+'.pth', map_location=torch.device(args.device)))
    else:    
        AE.load_state_dict(torch.load(args.output_model_dir + '/model_ae.pth', map_location=torch.device(args.device)))
        NN_PREDICTION.load_state_dict(torch.load(args.output_model_dir + '/model_nn.pth', map_location=torch.device(args.device)))

    AE.eval()
    NN_PREDICTION.eval()

    # run embedding
    mol_latent_train, mol_embedding_train, y_true_train, y_pred_train, ids_train, c_train = embed(loader_train, AE, NN_PREDICTION, device=args.device)
    # mol_latent_val, mol_embedding_val, y_true_val, y_pred_val, ids_val, c_val = embed(loader_val, AE, NN_PREDICTION, device=args.device)
    mol_latent_test, mol_embedding_test, y_true_test, y_pred_test, ids_test, c_test = embed(loader_test, AE, NN_PREDICTION, device=args.device)

    # evaluation on trained embedding
    print("#### Evaluation ####")
    X_train = np.array(mol_embedding_train.tolist())
    y_train_true = np.array(y_true_train.tolist())
    y_train_pred = np.array(y_pred_train.tolist())
    # X_val = np.array(mol_embedding_val.tolist())
    # y_val_true = np.array(y_true_val.tolist())
    # y_val_pred = np.array(y_pred_val.tolist())
    X_test = np.array(mol_embedding_test.tolist())
    y_test_true = np.array(y_true_test.tolist())
    y_test_pred = np.array(y_pred_test.tolist())

    df_plt = pd.DataFrame({'y_test':y_test_true, 'y_pred':y_test_pred})
    sns.set_style(style='ticks')
    color_1 = '#1C3077' # blue
    ax = sns.lmplot(x='y_test', y='y_pred', data=df_plt, 
                    scatter_kws={"s": 20, "color": color_1, "edgecolor": "none"}, 
                    line_kws={"color": color_1})
    ax.fig.set_dpi(300)
    plt.plot([0, 100], [0, 100], color='grey', linestyle='--', zorder=-1)
    plt.xlim((-10,110))
    plt.ylim((-10,110))
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.savefig(args.output_model_dir + '/scatter_plot.pdf', dpi=300)

    X_train = torch.tensor(mol_embedding_train.tolist())
    y_train = torch.tensor(y_true_train.tolist())
    # X_val = torch.tensor(mol_embedding_val.tolist())
    # y_val = torch.tensor(y_true_val.tolist())
    X_test = torch.tensor(mol_embedding_test.tolist())
    y_test = torch.tensor(y_test_true.tolist())
    y_test_true = torch.tensor(y_test_true.tolist())
    y_test_pred = torch.tensor(y_test_pred.tolist())

    with torch.no_grad():
        MSE = nn.MSELoss()
        mse = MSE(y_test_pred, y_test_true)
        mse = float(mse)
        print("MSE loss: ", mse) 

        MAE = nn.L1Loss() 
        mae = MAE(y_test_pred, y_test_true) 
        mae = float(mae)
        print("MAE loss: ", mae) 

    from torchmetrics.regression import R2Score
    with torch.no_grad():
        r2score = R2Score()
        r2 = r2score(y_test_pred, y_test_true)
        r2 = float(r2)
        print("R2 score: ", r2)

    from torchmetrics.regression import MeanSquaredError
    with torch.no_grad():
        mean_squared_error = MeanSquaredError(squared=False)
        rmse = mean_squared_error(y_test_pred, y_test_true)
        rmse = float(rmse)
        print("RMSE: ", rmse)

    # save result pretraining
    record = {'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE': rmse}
    hyper_list = ['embedding_type', 'encoder_type', 'decoder_type', 'batch_size', 'emb_dim', 'beta', 'alpha', 'num_layer', 'epochs', 'dropout_ratio', 'lr']
    with open('dataset/'+args.file + '/train_result.txt', "a") as f:
        f.write(str(args.seed)+'_'+args.time + '\t')
        for a in hyper_list:
            f.write(a + ': ' + str(vars(args)[a]) + '\t')
        for a in record:
            f.write(a + ': ' + str(round(record[a], 4)) + '\t')
        f.write('\n')


