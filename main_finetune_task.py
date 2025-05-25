# Library
import os
import time
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# catcvae
from catcvae.setup import ModelArgumentParser
from catcvae.molgraph import atom_encoder_m, bond_encoder_m, atom_decoder_m, bond_decoder_m, max_atom_number, matrix_size, matrix2mol
from catcvae.dataset import getDatasetFromFile, getDatasetObject, getDatasetSplittingFinetune, getDataLoader, getDataObject
from catcvae.condition import getConditionDim, getOneHotCondition
from catcvae.classweight import getClassWeight
from catcvae.loss import VAELoss, Annealer, recon_loss_fn, cosine_similarity, ALLLoss
from catcvae.ae import CVAE, latent_space_quality, sample_latent_space
from catcvae.training import save_model, save_loss, save_report, save_model_latest
from catcvae.prediction import NN, NN_TASK
from catcvae.latent import embed, save_latent

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
    else:
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
    # setup predictive model NN_TASK
    NN_PREDICTION = NN_TASK(in_dim=args.emb_dim+(3*args.emb_cond_dim)+args.cond_dim, out_dim_class=1).to(args.device)
    print(NN_PREDICTION)
    # setup loss function
    if args.annealing:
        annealing_agent = Annealer(total_steps=args.annealing_slope_length, shape=args.annealing_shape)
    else:
        annealing_agent = None
    # LOSSCLASS = VAELoss(recon_loss_fn(args.AE_loss, class_weights=args.class_weights), annealing_agent)
    LOSSCLASS = VAELoss(args.AE_loss, class_weights=args.class_weights, annealer=annealing_agent, device=args.device)
    print(LOSSCLASS)
    # loss_fn_nn = nn.L1Loss()
    # loss_fn_nn = nn.MSELoss()
    loss_fn_nn = nn.HuberLoss(delta=0.2)
    print(loss_fn_nn)
    ALLLOSS = ALLLoss(vae_starting=100, nn_starting=0)

    # setup parameters
    model_param_group = [{'params': AE.parameters(), 'lr': args.lr},
                        {'params': NN_PREDICTION.parameters(), 'lr': args.lr}]
    # setup optimizers
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = np.inf
    # setup learning rate scheduler
    # lr_scheduler = MultiStepLR(optimizer, 
    #                 milestones=[450, 480], 
    #                 gamma=0.05, verbose=False)

    # setup output directory
    args.time = time.strftime("%Y%m%d_%H%M%S")
    if args.name is not None: args.time+='_'+str(args.name)
    args.output_model_dir = 'dataset/'+args.file+'/output'+'_'+str(args.seed)+'_'+str(args.time)
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
        with open(args.output_model_dir + '/setup.txt', "a") as f:
            for a in vars(args):
                f.write(a + ' ' + str(vars(args)[a]) + '\n')

    # trained dataset folder 
    file_trained = args.pretrained_file
    seed_trained = 0 
    time_trained = args.pretrained_time
    output_model_dir_trained = 'dataset/'+file_trained+'/output'+'_'+str(seed_trained)+'_'+time_trained
    epoch_selected = None
    if epoch_selected is not None:
        AE.load_state_dict(torch.load(output_model_dir_trained + '/model_ae_'+str(epoch_selected)+'.pth', map_location=torch.device(args.device)))
        # NN_PREDICTION.load_state_dict(torch.load(output_model_dir_trained + '/model_nn_'+str(epoch_selected)+'.pth', map_location=torch.device(args.device)))
    else:    
        AE.load_state_dict(torch.load(output_model_dir_trained + '/model_ae.pth', map_location=torch.device(args.device)))
        # NN_PREDICTION.load_state_dict(torch.load(output_model_dir_trained + '/model_nn.pth', map_location=torch.device(args.device)))

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
            # all_loss['nn_loss'] = nn_loss
            # loss = AE_loss+nn_loss
            loss, all_loss_all = ALLLOSS(AE_loss, nn_loss)
            all_loss['nn_loss'] = all_loss_all['nn_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # AE_loss_accum += AE_loss.detach().cpu().item()+nn_loss.detach().cpu().item()
            AE_loss_accum += loss.detach().cpu().item()
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
        # temp_loss = all_loss_val['nn_loss']
        if temp_loss < optimal_loss and ALLLOSS.saveStart():
            optimal_loss = temp_loss
            save_model(args.output_model_dir, optimal_loss, AE, nn=NN_PREDICTION)
            # save_latent(args.output_model_dir, loader_all, AE, NN_PREDICTION)

        # if epoch%50 == 0 or epoch == args.epochs-1 or epoch in [1, 2, 3, 4, 10]:
        # if epoch == args.epochs-1:
            # save_model(args.output_model_dir, temp_loss, AE, nn=NN_PREDICTION, save_best=False, epoch=epoch)
            # save_latent(args.output_model_dir, loader_all, AE, NN_PREDICTION, epoch, device=args.device)
        
        sample_num = 10
        corr, unique, sample_smiles = latent_space_quality(AE, ae_type=args.AE_type, sample_num=sample_num, datasets_dobj=datasets_dobj_train, device=args.device)
        validity = corr * 100. / sample_num
        diversity = unique * 100. / sample_num
        # save to text
        with open(args.output_model_dir + '/sample_smiles.txt', "w") as f:
            f.write('epoch: '+str(epoch)+'\n')
            f.write('validity: '+str(validity)+'\n')
            f.write('diversity:'+str(diversity)+'\n')
            for s in sample_smiles:
                f.write(s+'\n')
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
        ALLLOSS.step()
        # lr_scheduler.step()

        return 

    # start training
    for epoch in tqdm(range(args.epochs)):
        print('epoch: {}'.format(epoch))
        train_model(args, loader_train, loader_val, optimizer, epoch)

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
    # mol_latent_val, mol_embedding_val, y_true_val, y_pred_val, ids_val, c_val = embed(loader_val)
    mol_latent_test, mol_embedding_test, y_true_test, y_pred_test, ids_test, c_test = embed(loader_test, AE, NN_PREDICTION, device=args.device)

    # evaluation on trained embedding
    print("#### Evaluation ####")
    X_train = torch.tensor(mol_embedding_train.tolist())
    y_train_true = torch.tensor(y_true_train.tolist())
    y_train_pred = torch.tensor(y_pred_train.tolist())
    # X_val = torch.tensor(mol_embedding_val.tolist())
    # y_val_true = torch.tensor(y_true_val.tolist())
    # y_val_pred = torch.tensor(y_pred_val.tolist())
    X_test = torch.tensor(mol_embedding_test.tolist())
    y_test_true = torch.tensor(y_true_test.tolist())
    y_test_pred = torch.tensor(y_pred_test.tolist())

    print("TRAIN EVALUATION")
    with torch.no_grad():
        MSE = nn.MSELoss()
        mse = MSE(y_train_pred, y_train_true)
        mse = float(mse)
        print("MSE: ", mse) 

        MAE = nn.L1Loss() 
        mae = MAE(y_train_pred, y_train_true) 
        mae = float(mae)
        print("MAE: ", mae)
    
    from torchmetrics.regression import R2Score
    with torch.no_grad():
        r2score = R2Score()
        r2 = r2score(y_train_pred, y_train_true)
        r2 = float(r2)
        print("R2: ", r2)

    from torchmetrics.regression import MeanSquaredError
    with torch.no_grad():
        mean_squared_error = MeanSquaredError(squared=False)
        rmse = mean_squared_error(y_train_pred, y_train_true)
        rmse = float(rmse)
        print("RMSE: ", rmse)

    df_plt = pd.DataFrame({'y_train':y_train_true, 'y_pred_train':y_train_pred})
    sns.set_style(style='ticks')
    color_1 = '#1C3077' # blue
    ax = sns.lmplot(x='y_train', y='y_pred_train', data=df_plt, 
                    scatter_kws={"s": 20, "color": color_1, "edgecolor": "none"}, 
                    line_kws={"color": color_1})
    ax.fig.set_dpi(300)
    plt.plot([min(y_test_true), max(y_test_true)], [min(y_test_true), max(y_test_true)], color='grey', linestyle='--', zorder=-1)
    maxmin = max(y_test_true)-min(y_test_true)
    plt.text(5, 90, 'R$^2$ = %.2f' % r2, fontsize=14)
    plt.xlim((min(y_test_true)-0.1*maxmin, max(y_test_true)+0.1*maxmin))
    plt.ylim((min(y_test_true)-0.1*maxmin, max(y_test_true)+0.1*maxmin))
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.show()
    plt.savefig(args.output_model_dir + '/scatter_plot_train.pdf', dpi=300)

    print("TEST EVALUATION")
    with torch.no_grad():
        MSE = nn.MSELoss()
        mse = MSE(y_test_pred, y_test_true)
        mse = float(mse)
        print("MSE: ", mse) 

        MAE = nn.L1Loss() 
        mae = MAE(y_test_pred, y_test_true) 
        mae = float(mae)
        print("MAE: ", mae) 

    from torchmetrics.regression import R2Score
    with torch.no_grad():
        r2score = R2Score()
        r2 = r2score(y_test_pred, y_test_true)
        r2 = float(r2)
        print("R2: ", r2)

    from torchmetrics.regression import MeanSquaredError
    with torch.no_grad():
        mean_squared_error = MeanSquaredError(squared=False)
        rmse = mean_squared_error(y_test_pred, y_test_true)
        rmse = float(rmse)
        print("RMSE: ", rmse)

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
    sns.set_theme(font='Helvetica')
    color_1 = '#1C3077' # blue
    ax = sns.lmplot(x='y_test', y='y_pred', data=df_plt, 
                    scatter_kws={"s": 20, "color": color_1, "edgecolor": "none"}, 
                    line_kws={"color": color_1})
    ax.fig.set_dpi(300)
    ax.fig.set_size_inches(4, 4)
    plt.plot([min(y_test_true), max(y_test_true)], [min(y_test_true), max(y_test_true)], color='grey', linestyle='--', zorder=-1)
    maxmin = max(y_test_true)-min(y_test_true)
    plt.text(5, 90, 'R$^2$ = %.2f' % r2, fontsize=14)
    plt.xlim((min(y_test_true)-0.1*maxmin, max(y_test_true)+0.1*maxmin))
    plt.ylim((min(y_test_true)-0.1*maxmin, max(y_test_true)+0.1*maxmin))
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.show()
    plt.savefig(args.output_model_dir + '/scatter_plot.pdf', dpi=300)

    # Evaluation using trained embedding for surrogate regression
    print("#### RandomForestRegressor ####")
    model = RandomForestRegressor(n_estimators=300, max_depth=5)
    model.fit(X_train, y_train_true)
    r_sq = model.score(X_train, y_train_true)
    print(f"Train: R2: {r_sq}")

    # Evaluating the trained model on training data
    y_pred = model.predict(X_train)
    # y_pred = np.clip(y_pred, 0, 100)
    r_sq = model.score(X_train, y_train_true)
    print(f"Train: R2: {r_sq}")
    print("Train: MAE:" , metrics.mean_absolute_error(y_train_true, y_pred))
    # Evaluating the trained model on test data
    y_pred = model.predict(X_test)
    # y_pred = np.clip(y_pred, 0, 100)
    r_sq = model.score(X_test, y_test_true)
    print(f"Test: R2: {r_sq}")
    print("Test: MAE:" , metrics.mean_absolute_error(y_test_true, y_pred))

    sns.set_style(style='ticks')
    color_1 = '#1C3077' # blue
    ax = sns.lmplot(x='y_test', y='y_pred', data=pd.DataFrame({'y_test':y_test_true, 'y_pred':y_pred}), 
                    scatter_kws={"s": 20, "color": color_1, "edgecolor": "none"}, 
                    line_kws={"color": color_1})
    ax.fig.set_dpi(300)
    plt.plot([min(y_test_true), max(y_test_true)], [min(y_test_true), max(y_test_true)], color='grey', linestyle='--', zorder=-1)
    plt.text(5, 90, 'R$^2$ = %.2f' % r_sq, fontsize=14)
    maxmin = max(y_test_true)-min(y_test_true)
    plt.xlim((min(y_test_true)-0.1*maxmin, max(y_test_true)+0.1*maxmin))
    plt.ylim((min(y_test_true)-0.1*maxmin, max(y_test_true)+0.1*maxmin))
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.show()
    plt.savefig(args.output_model_dir + '/scatter_plot_rfr.pdf', dpi=300)

    # save result hyperparameter
    record = {'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE': rmse, 'R2_rfr': r_sq}
    hyper_list = ['embedding_type', 'encoder_type', 'decoder_type', 'batch_size', 'emb_dim', 'beta', 'alpha', 'num_layer', 'epochs', 'dropout_ratio', 'lr']
    with open('dataset/'+args.file + '/hyper_result.txt', "a") as f:
        f.write(str(seed_trained)+'_'+time_trained + '\t')
        f.write(str(args.seed)+'_'+args.time + '\t')
        for a in hyper_list:
            f.write(a + ': ' + str(vars(args)[a]) + '\t')
        for a in record:
            f.write(a + ': ' + str(round(record[a], 4)) + '\t')
        f.write('\n')
    