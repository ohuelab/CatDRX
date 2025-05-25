import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from catcvae.molgraph import max_atom_number, matrix_size, matrix2mol
from catcvae.condition import getSampleCondition
from catcvae.embedding.gnn import GNN
from catcvae.encoder.matrixencoder import MATRIXENCODER
from catcvae.decoder.matrixdecoder import MATRIXDECODER
from catcvae.latent import embed
from catcvae.molgraph import atom_encoder_m, bond_encoder_m
len_atom_type = len(atom_encoder_m) 
len_bond_type = len(bond_encoder_m)

class CVAE(torch.nn.Module):
    def __init__(self, embedding_setting, encoding_setting, decoding_setting, emb_dim, emb_cond_dim, cond_dim, device='cpu'):
        super(CVAE, self).__init__()

        self.device = device
        self.embedding_setting = embedding_setting
        self.encoding_setting = encoding_setting
        self.decoding_setting = decoding_setting

        # EMBEDDING for CATALYST
        if embedding_setting['type'] == 'GNN': # ENCODING must be None
            self.embedding_catalyst = GNN(num_layer=embedding_setting['num_layer'], 
                                        emb_dim=embedding_setting['emb_dim'], 
                                        JK=embedding_setting['JK'], 
                                        readout=embedding_setting['readout'], 
                                        dropout_ratio=embedding_setting['dropout_ratio'], 
                                        gnn_type=embedding_setting['gnn_type'],
                                        device=self.device).to(self.device)
        elif embedding_setting['type'] == 'None': # ENCODING must be MATRIX
            self.embedding_catalyst = None

        # EMBEDDING for CONDITION
        self.embedding_reactant = GNN(num_layer=embedding_setting['num_layer'], 
                                    emb_dim=embedding_setting['emb_cond_dim'], 
                                    JK=embedding_setting['JK'], 
                                    readout=embedding_setting['readout'], 
                                    dropout_ratio=embedding_setting['dropout_ratio'], 
                                    gnn_type=embedding_setting['gnn_type'],
                                    device=self.device).to(self.device)
        self.embedding_reagent = GNN(num_layer=embedding_setting['num_layer'], 
                                    emb_dim=embedding_setting['emb_cond_dim'], 
                                    JK=embedding_setting['JK'], 
                                    readout=embedding_setting['readout'], 
                                    dropout_ratio=embedding_setting['dropout_ratio'], 
                                    gnn_type=embedding_setting['gnn_type'],
                                    device=self.device).to(self.device)
        self.embedding_product = GNN(num_layer=embedding_setting['num_layer'], 
                                    emb_dim=embedding_setting['emb_cond_dim'], 
                                    JK=embedding_setting['JK'], 
                                    readout=embedding_setting['readout'], 
                                    dropout_ratio=embedding_setting['dropout_ratio'], 
                                    gnn_type=embedding_setting['gnn_type'],
                                    device=self.device).to(self.device)
        self.cond_dim = cond_dim
        self.emb_cond_dim = emb_cond_dim
        self.emb_dim = emb_dim

        # ENCODER for CATALYST
        if encoding_setting['type'] == 'MATRIX': # EMBEDDING must be None
            self.encoder = MATRIXENCODER(input_size=encoding_setting['emb_dim'], # emb_dim+cond_dim
                                         max_atom_number=decoding_setting['max_atom_number'], # max_atom_number
                                         len_atom_type=decoding_setting['len_atom_type'], # len_atom_type
                                         len_bond_type=decoding_setting['len_bond_type'], # len_bond_type
                                         matrix_size=encoding_setting['matrix_size'], # matrix_size
                                         num_layers=encoding_setting['num_layer'],
                                         dropout_ratio=encoding_setting['dropout_ratio'],
                                         device=self.device).to(self.device)
        elif encoding_setting['type'] == 'None': # EMBEDDING must be GNN
            self.encoder = None

        self.fc_mu = nn.Linear(self.emb_dim+(3*self.emb_cond_dim)+self.cond_dim, self.emb_dim).to(self.device)
        self.fc_logvar = nn.Linear(self.emb_dim+(3*self.emb_cond_dim)+self.cond_dim, self.emb_dim).to(self.device)

        # DECODER for CATALYST
        if decoding_setting['type'] == 'MATRIX':
            self.decoder = MATRIXDECODER(input_size=decoding_setting['emb_dim']+(3*decoding_setting['emb_cond_dim'])+decoding_setting['cond_dim'], # emb_dim+cond_dim
                                         max_atom_number=decoding_setting['max_atom_number'], # max_atom_number
                                         len_atom_type=decoding_setting['len_atom_type'], # len_atom_type
                                         len_bond_type=decoding_setting['len_bond_type'], # len_bond_type
                                         matrix_size=decoding_setting['matrix_size'], # matrix_size
                                         num_layers=decoding_setting['num_layer'],
                                         dropout_ratio=decoding_setting['dropout_ratio'],
                                         teacher_forcing=decoding_setting['teacher_forcing'],
                                         device=self.device).to(self.device)

    # EMBED
    def embed(self, batch, current_batch_size):
        if self.embedding_catalyst is not None:
            if self.embedding_setting['type'] == 'GNN':
                node_feat_catalyst, edge_index_catalyst, edge_attr_catalyst, batch_batch_catalyst = batch.x_catalyst, batch.edge_index_catalyst, batch.edge_attr_catalyst, batch.x_catalyst_batch
                node_representation_catalyst, graph_representation_catalyst = self.embedding_catalyst(node_feat_catalyst, edge_index_catalyst, edge_attr_catalyst, batch_batch_catalyst)
                x = graph_representation_catalyst
                assert torch.isnan(graph_representation_catalyst).sum() == 0, 'graph_representation_catalyst is NaN'
        else:
            if self.encoding_setting['type'] == 'MATRIX':
                x = batch.matrix_catalyst.view(current_batch_size, -1).to(self.device)
        
        if self.decoding_setting['type'] == 'MATRIX':
            x_target = batch.matrix_catalyst.view(current_batch_size, -1).to(self.device)
            target_tensor = batch.matrix_catalyst.view(current_batch_size, -1).to(self.device)
            guide_tensor = None
        
        return x, x_target, target_tensor, guide_tensor
    
    # CONDITION EMBED
    def condition(self, batch):
        # reactant
        node_feat_reactant, edge_index_reactant, edge_attr_reactant, batch_batch_reactant = batch.x_reactant, batch.edge_index_reactant, batch.edge_attr_reactant, batch.x_reactant_batch
        node_representation_reactant, graph_representation_reactant = self.embedding_reactant(node_feat_reactant, edge_index_reactant, edge_attr_reactant, batch_batch_reactant)
        assert torch.isnan(graph_representation_reactant).sum() == 0, 'graph_representation_reactant is NaN'
        # reagent
        node_feat_reagent, edge_index_reagent, edge_attr_reagent, batch_batch_reagent = batch.x_reagent, batch.edge_index_reagent, batch.edge_attr_reagent, batch.x_reagent_batch
        node_representation_reagent, graph_representation_reagent = self.embedding_reagent(node_feat_reagent, edge_index_reagent, edge_attr_reagent, batch_batch_reagent)
        assert torch.isnan(graph_representation_reagent).sum() == 0, 'graph_representation_reagent is NaN'
        # product
        node_feat_product, edge_index_product, edge_attr_product, batch_batch_product = batch.x_product, batch.edge_index_product, batch.edge_attr_product, batch.x_product_batch
        node_representation_product, graph_representation_product = self.embedding_product(node_feat_product, edge_index_product, edge_attr_product, batch_batch_product)
        assert torch.isnan(graph_representation_product).sum() == 0, 'graph_representation_product is NaN'
        c_graph = torch.cat((graph_representation_reactant, graph_representation_reagent, graph_representation_product), dim=1)
        # with other conditions
        c = torch.cat((c_graph, batch.c), dim=1)
        return c
    
    # ENCODE
    def encode(self, x, c):
        if self.encoder is not None:
            x = self.encoder(x)
        xc = torch.cat((x, c), dim=1)
        mu = self.fc_mu(xc)
        logvar = self.fc_logvar(xc)
        return mu, logvar

    # REPARAMETERIZE
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z
        
    # DECODE
    def decode(self, z, c, target_tensor=None, guide_tensor=None):
        zc = torch.cat((z, c), dim=1)
        y_hat = self.decoder(zc, target_tensor, guide_tensor)
        return y_hat

    def forward(self, batch):
        # Embed catalyst
        x, x_target, target_tensor, guide_tensor = self.embed(batch, current_batch_size=len(batch.id))
        # Embed condition
        c = self.condition(batch)
        assert torch.isnan(x).sum() == 0, 'x is NaN'
        assert torch.isnan(c).sum() == 0, 'c is NaN'
        # Encode catalyst+condition
        mu, logvar = self.encode(x, c)
        assert torch.isnan(mu).sum() == 0, 'mu is NaN'
        assert torch.isnan(logvar).sum() == 0, 'logvar is NaN'
        z = self.reparameterize(mu, logvar)
        assert torch.isnan(z).sum() == 0, 'z is NaN'
        # Decode
        y_decoded = self.decode(z, c, target_tensor, guide_tensor)
        # assert torch.isnan(y_decoded).sum() == 0, 'y_decoded is NaN'
        
        return x_target, y_decoded, mu, logvar, z, c
    

def sample_latent_space(ae, c, mol_latent=None, target_tensor=None, guide_tensor=None, noise=False, dim_range=None, device='cpu'):
    """
    Samples a point from the latent space and decodes it to generate a molecule.

    Args:
        ae (CVAE): The conditional variational autoencoder model.
        c (torch.Tensor): The condition tensor.
        mol_latent (torch.Tensor, optional): A sample latent point (e.g. from training set). Defaults to None (random sampling).
        target_tensor (torch.Tensor, optional): Target or ground-truth tensor. Defaults to None.
        guide_tensor (torch.Tensor, optional): Guide tensor for decoding. Defaults to None.
        noise (bool, optional): Whether to add noise to the latent point. Defaults to False.
        dim_range (list of tuples, optional): Range for each dimension of the latent space (e.g. training space). Defaults to None.
        device (str, optional): The device to run the computation on. Defaults to 'cpu'.

    Returns:
        tuple: A tuple containing:
            - out_matrix (torch.Tensor): The decoded molecular matrix (ready to reconstruction into molecule).
            - out_decoded (torch.Tensor): The raw decoded output (not yet reshaped).
            - latent_point (torch.Tensor): The sampled latent point.
    """
    ae.eval()
    
    with torch.no_grad():
        if mol_latent is None and target_tensor is None:
            if dim_range is not None:
                low = torch.tensor([d[0] for d in dim_range], dtype=torch.float, device=device)
                high = torch.tensor([d[1] for d in dim_range], dtype=torch.float, device=device)
                latent_point = low + (high - low) * torch.rand(1, ae.emb_dim, device=device)
            else:
                latent_point = torch.randn(1, ae.emb_dim, device=device)
        else:
            latent_point = torch.tensor(mol_latent, device=device).unsqueeze(0)
            if noise:
                noise = torch.normal(mean=0, std=1, size=(1, ae.emb_dim), device=device)
                latent_point = latent_point + noise

        out_decoded = ae.decode(latent_point, c, target_tensor=target_tensor, guide_tensor=guide_tensor)
        out_matrix = out_decoded.reshape(max_atom_number, matrix_size//max_atom_number)
    
    # ae.train()
    return out_matrix, out_decoded, latent_point

    
def latent_space_quality(ae, ae_type, sample_num, datasets_dobj, device='cpu'):
    ae.eval()

    with torch.no_grad():
        total_correct = 0
        all_correct_molecules = set()

        # sample condition only from training set
        for _ in range(sample_num//2):
            t = random.randint(0, len(datasets_dobj)-1)
            mol_t = datasets_dobj[t]
            loader_mol_t = DataLoader([mol_t], batch_size=1, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
            c = ae.condition(next(iter(loader_mol_t)).to(device))
            out_matrix, out_decoded, latent_point = sample_latent_space(ae, c=c, device=device)
            mol = matrix2mol(out_matrix)
            if mol:
                try:
                    mol.UpdatePropertyCache(strict=False)
                    smiles = Chem.MolToSmiles(mol)
                except Exception:
                    smiles = ""
                    np.savez('error_out_matrix.npz', out_matrix=out_matrix.cpu())
                if is_correct_smiles(smiles):
                    total_correct += 1
                    all_correct_molecules.add(smiles)
                else:
                    np.savez('error_out_matrix.npz', out_matrix=out_matrix.cpu())
        
        # sample latent point+condition from training set
        for _ in range(sample_num//2):
            t = random.randint(0, len(datasets_dobj)-1)
            mol_t = datasets_dobj[t]
            loader_mol_t = DataLoader([mol_t], batch_size=1, shuffle=False, follow_batch=['x_reactant', 'x_reagent', 'x_product', 'x_catalyst'])
            mol_latent_t, mol_embedding_t, y_true_t, y_pred_t, ids_t, c_t = embed(loader_mol_t, ae, None, device=device)
            # c = ae.condition(next(iter(loader_mol_t)).to(device))
            c_t = torch.tensor(c_t, device=device)
            out_matrix, out_decoded, latent_point = sample_latent_space(ae, c=c_t, mol_latent=mol_latent_t.squeeze(), device=device)
            mol = matrix2mol(out_matrix)
            if mol:
                try:
                    mol.UpdatePropertyCache(strict=False)
                    smiles = Chem.MolToSmiles(mol)
                except Exception:
                    smiles = ""
                    np.savez('error_out_matrix.npz', out_matrix=out_matrix.cpu())
                if is_correct_smiles(smiles):
                    total_correct += 1
                    all_correct_molecules.add(smiles)
                else:
                    np.savez('error_out_matrix.npz', out_matrix=out_matrix.cpu())
        
    ae.train()
    return total_correct, len(all_correct_molecules), list(all_correct_molecules)


def is_correct_smiles(smiles):
    if smiles == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
        return mol is not None
    except Exception:
        return False