import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def embed(loader, AE, NN_PREDICTION=None, device='cpu'):
    """
    Extracts latent representations, embeddings, and other relevant information
    from the autoencoder (CVAE) and predictor (NN_PREDICTION).

    Args:
        loader: DataLoader providing batches of data.
        AE: Autoencoder model.
        NN_PREDICTION: Neural network predictor for predictions. (optional)
        device: Device to run computations on ('cpu' or 'cuda').

    Returns:
        latent: Latent representations (z).
        embeddings: Latent embeddings (mu).
        y_true: True labels.
        y_pred: Predicted labels (if NN_PREDICTION is provided).
        ids: Identifiers for the data points.
        condition: Condition vectors (c).
    """
    AE.eval()
    if NN_PREDICTION is not None:
        NN_PREDICTION.eval()

    latent = []
    embeddings = []
    y_true = []
    y_pred = []
    ids = []
    condition = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # perform forward pass
            (x_target, y_decoded, mu, varlog, z, c) = AE(batch)
            if NN_PREDICTION is not None:
                if hasattr(batch, 'condition_extra'): # for containing extra conditions
                    y_prediction = NN_PREDICTION(mu, c, c_extra=batch.condition_extra)
                else:
                    y_prediction = NN_PREDICTION(mu, c)

            # latent vector
            latent.extend(z.detach().cpu().numpy()) 
            # latent embedding from mean
            embeddings.extend(mu.detach().cpu().numpy()) 
            # true labels
            y_true.extend(batch.y.detach().cpu().numpy()) 
            # predictions
            if NN_PREDICTION is not None:
                try:
                    y_pred.extend(torch.squeeze(y_prediction).detach().cpu().numpy())
                except:
                    y_pred.append(torch.squeeze(y_prediction).detach().cpu().numpy())
            # ids
            ids.extend(batch.id.detach().cpu() if isinstance(batch.id, torch.Tensor) else batch.id)
            # condition
            condition.extend(c.detach().cpu().numpy())

        return np.array(latent), np.array(embeddings), np.array(y_true), np.array(y_pred), np.array(ids), np.array(condition)

# record latent representation during training
def save_latent(output_model_dir, loader, AE, NN_PREDICTION, epoch_selected=None, device='cpu'):
    mol_latent_all, mol_embedding_all, y_true_all, y_pred_all, ids_all, c_all = embed(loader, AE, NN_PREDICTION, device=device)

    mol_embedding_df = pd.DataFrame(mol_embedding_all.tolist())
    mol_embedding_df['y'] = y_true_all.tolist()
    mol_embedding_df['ids'] = ids_all
    
    data_subset = mol_embedding_df.drop(columns=['y', 'ids'])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_subset)
    mol_embedding_df['pca-one'] = pca_result[:,0]
    mol_embedding_df['pca-two'] = pca_result[:,1] 
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    mol_embedding_df['tsne-2d-one'] = tsne_results[:,0]
    mol_embedding_df['tsne-2d-two'] = tsne_results[:,1]

    sns.set_theme(style="white", palette=None)

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
        plt.savefig(output_model_dir+'/embedding_'+str(hue_column)+'_'+str(epoch_selected)+'.png')
    else:
        plt.savefig(output_model_dir+'/embedding_'+str(hue_column)+'.png')