
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from catcvae.molgraph import atom_encoder_m, bond_encoder_m, max_atom_number, matrix_size, matrix2mol
len_atom_type = len(atom_encoder_m) 
len_bond_type = len(bond_encoder_m)


def cosine_similarity(p, z, average=True):
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    loss = -(p * z).sum(dim=1)
    loss = (loss + 1)
    if average:
        loss = loss.mean()
    return loss

def recon_loss_fn(loss_type, class_weights=None):
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'l2':
        return nn.MSELoss(reduction="sum")
    elif loss_type == 'BCE':
        return nn.BCELoss(reduction="mean")
    elif loss_type == 'BCEWithLogits':
        return nn.BCEWithLogitsLoss(reduction="mean", pos_weight=class_weights)
    elif loss_type == 'CE':
        return nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    elif loss_type == 'cosine':
        return cosine_similarity

# ref: https://github.com/hubertrybka/vae-annealing
class VAELoss(torch.nn.Module):
    """
    Calculates reconstruction loss and KL divergence loss for VAE.
    """
    def __init__(self, fn, class_weights=None, annealer=None, device='cpu'):
        """
        Args:
            fn (str): loss function to use for reconstruction loss. Options: 'l1', 'l2', 'BCE', 'BCEWithLogits', 'CE', 'cosine'.
            class_weights (dict): dictionary containing class weights for each component in matrix of the loss function.
            annealer (Annealer): Annealer object for KL divergence loss.
            device (str): device to use for computation ('cpu' or 'cuda').
        """
        super(VAELoss, self).__init__()
        if class_weights is None:
            class_weights = {'atom': None, 'annotation': None, 'adjacency': None}
        self.loss_fn_atom = recon_loss_fn(fn, class_weights=class_weights['atom'])
        self.loss_fn_annotation = recon_loss_fn(fn, class_weights=class_weights['annotation'])
        self.loss_fn_adjacency = recon_loss_fn(fn, class_weights=class_weights['adjacency'])
        self.annealing_agent = annealer

    def forward(self, y_decoded, y, mu, logvar, alpha=1, beta=1, device='cpu'):
        """
        Args:
            y_decoded (torch.Tensor): reconstructed input matrix
            y (torch.Tensor): original input matrix
            mu (torch.Tensor): latent space mu
            logvar (torch.Tensor): latent space log variance
            alpha (float): weight for reconstruction loss
            beta (float): weight for KL divergence loss
            device (str): device to use for computation ('cpu' or 'cuda')
        Returns:
            loss (torch.Tensor): total loss
            {'recon_loss': recon_loss, 'kl_loss': kl_loss}: dictionary containing reconstruction loss and KL divergence loss
        """
        # reconstruction loss all
        # recon_loss = self.loss_fn(y_decoded.to(device), y.to(device))

        # reconstruction loss each component
        y = y.reshape(-1, max_atom_number, matrix_size//max_atom_number).to(device) # [b, N, A+N*B]
        # define mask based on true length column
        y_mask = y[:, :, 1:len_atom_type+1].reshape(-1, max_atom_number, len_atom_type).argmax(dim=2).reshape(-1).to(device) # [b*N]
        y_mask = (y_mask != 0).type(torch.bool).to(device) # [b*N]
        
        # true length column
        y_max_atom_number_output = y[:, :, 0].argmax(dim=1).to(device) # [b]
        # true annotation_matrix
        y_annotation_matrix_output = y[:, :, 1:len_atom_type+1].reshape(-1, max_atom_number, len_atom_type) # [b, N, A]
        y_annotation_matrix_output = y_annotation_matrix_output.argmax(dim=2) # [b, N]
        y_annotation_matrix_output = y_annotation_matrix_output.reshape(-1)[y_mask].to(device) # [b*N]
        # print("#1: true annotation_matrix", y_annotation_matrix_output.shape)
        # true adjacency_matrix
        y_adjacency_matrix_output = y[:, :, len_atom_type+1:].reshape(-1, max_atom_number, len_bond_type) # [b*N, N, B]
        y_adjacency_matrix_output = y_adjacency_matrix_output.argmax(dim=2) # [b*N, N]
        y_adjacency_matrix_output = y_adjacency_matrix_output[y_mask].reshape(-1).to(device) # [b*N*N]
        # print("#2: true adjacency_matrix", y_adjacency_matrix_output.shape)
        
        y_decoded = y_decoded.reshape(-1, max_atom_number, matrix_size//max_atom_number).to(device) # [b, N, A+N*B]
        # decoded length column
        y_decoded_max_atom_number_output = y_decoded[:, :, 0].to(device) # [b, N]
        # decoded annotation_matrix
        y_decoded_annotation_matrix_output = y_decoded[:, :, 1:len_atom_type+1].reshape(-1, len_atom_type) # [b*N, A]
        y_decoded_annotation_matrix_output = y_decoded_annotation_matrix_output[y_mask].to(device) # [b*N, A]
        # print("#3: decoded annotation_matrix", y_decoded_annotation_matrix_output.shape) 
        # decoded adjacency_matrix
        y_decoded_adjacency_matrix_output = y_decoded[:, :, len_atom_type+1:].reshape(-1, max_atom_number*len_bond_type) # [b*N, N*B]
        y_decoded_adjacency_matrix_output = y_decoded_adjacency_matrix_output[y_mask].reshape(-1, len_bond_type).to(device) # [b*N*N, B]
        # print("#4: decoded adjacency_matrix", y_decoded_adjacency_matrix_output.shape)
        
        recon_loss = self.loss_fn_atom(y_decoded_max_atom_number_output, y_max_atom_number_output) + \
                     self.loss_fn_annotation(y_decoded_annotation_matrix_output, y_annotation_matrix_output) + \
                     self.loss_fn_adjacency(y_decoded_adjacency_matrix_output, y_adjacency_matrix_output)

        assert torch.isnan(recon_loss).sum() == 0, 'Reconstruction loss is NaN'
        
        # kl loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        assert torch.isnan(kl_loss).sum() == 0, 'KL (before annealing) loss is NaN'
        if self.annealing_agent is not None:
            kl_loss = self.annealing_agent(kl_loss)
        assert torch.isnan(kl_loss).sum() == 0, 'KL (after annealing) loss is NaN'
        
        loss = alpha * recon_loss + beta * kl_loss
        return loss, {'recon_loss': alpha * recon_loss, 'kl_loss': beta * kl_loss}
    

class ALLLoss(torch.nn.Module):
    def __init__(self, vae_starting=0, nn_starting=0):
        """
        Args:
            vae_starting (int): Number of epochs to start VAE training loss.
            nn_starting (int): Number of epochs to start NN training loss.
        """
        super(ALLLoss, self).__init__()
        self.vae_starting = vae_starting
        self.nn_starting = nn_starting
    
    def forward(self, vae_loss, nn_loss):
        if self.vae_starting == 0 and self.nn_starting == 0:
            loss = vae_loss + nn_loss
        elif self.vae_starting == 0 and self.nn_starting > 0:
            loss = vae_loss
        elif self.vae_starting > 0 and self.nn_starting == 0:
            loss = nn_loss
        elif self.vae_starting > 0 and self.nn_starting > 0:
            loss = 0.00
        # print("vae_loss: ", vae_loss.item())
        # print("nn_loss: ", nn_loss.item())
        return loss, {'vae_loss': vae_loss, 'nn_loss': nn_loss}

    # countdown vae_starting and nn_starting
    def step(self):
        if self.vae_starting > 0:
            self.vae_starting -= 1
        if self.nn_starting > 0:
            self.nn_starting -= 1
        return
    
    # check if both vae_starting and nn_starting are 0 (ready to save model)
    def saveStart(self):
        return self.vae_starting == 0 and self.nn_starting == 0

# ref: https://github.com/hubertrybka/vae-annealing
class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return