import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cos_distance
import rdkit
import rdkit.Chem as Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from catcvae.utils import mol_to_smiles, smiles_to_mol
from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
    

def validity(smiles_list, num_samples=0, verbose=False):
    valid = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid += 1
    validity_value = valid / num_samples if num_samples > 0 else 0
    if verbose:
        print(f"validity: {valid}/{num_samples}, {validity_value:.4f}")
    return validity_value


def uniqueness(smiles_list, verbose=False):
    smiles_list_new = []
    for smiles in smiles_list:
        try:
            smiles_list_new.append(Chem.CanonSmiles(smiles, useChiral=False))
        except:
            smiles_list_new.append(smiles)
    uniqueness_value = len(set(smiles_list_new)) / len(smiles_list_new) if len(smiles_list_new) > 0 else 0
    if verbose:
        print(f"uniqueness: {len(set(smiles_list_new))}/{len(smiles_list_new)}, {uniqueness_value:.4f}")
    return uniqueness_value


def novelty(smiles_list, ref_list, verbose=False):
    novel = 0
    smiles_list_new = []
    for smiles in smiles_list:
        try:
            smiles_list_new.append(Chem.CanonSmiles(smiles, useChiral=False))
        except:
            smiles_list_new.append(smiles)
    ref_list_new = []
    for smiles in ref_list:
        try:
            ref_list_new.append(Chem.CanonSmiles(smiles, useChiral=False))
        except:
            ref_list_new.append(smiles)
    for smiles in smiles_list_new:
        if smiles not in ref_list_new:
            novel += 1
    novelty_value = novel / len(smiles_list_new) if len(smiles_list_new) > 0 else 0
    if verbose:
        print(f"novelty: {novel}/{len(smiles_list_new)}, {novelty_value:.4f}")
    return novelty_value


def get_valid_and_unique(smiles_list):
    result = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        try:
            smiles = Chem.CanonSmiles(smiles, useChiral=False)
        except:
            pass
        if mol is not None:
            if smiles not in result:
                result.append(smiles)
    return result


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


def internal_diversity(smiles_list, radius=2, dictionary=None):
    diversity_list = []
    for i, a in tqdm(enumerate(smiles_list)):
        for b in smiles_list[i+1:]:
            diversity_list.append(1 - similarity(a, b, radius=radius, dictionary=dictionary))
    return np.mean(diversity_list), np.std(diversity_list)


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


def frechet_distance(smiles_list, ref_list):
    # Load chemnet model
    model = load_ref_model()
    # get canonical smiles and filter invalid ones
    can_smiles = [w for w in canonical_smiles(smiles_list) if w is not None]
    can_ref = [w for w in canonical_smiles(ref_list) if w is not None]
    # get predictions
    fcd_score = get_fcd(can_smiles, can_ref, model)
    return fcd_score
