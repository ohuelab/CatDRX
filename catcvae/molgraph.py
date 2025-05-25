# standard
import os as os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import torch
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
# import deepchem.feat as dcfeat
from catcvae.utils import *

# Updated
# 1: added (reaxys_homo)
# + add more element 60->66
# + add more bond 4->5 (Chem.rdchem.BondType.DATIVE)
# + change fn GetDegree one_of_k_encoding->one_of_k_encoding_unk (>10)
# + change num_node_features 94->100
# + change num_edge_features 10->11
# 2: added (ord_surf)
# + change num_node_features 100->101

ATOM_VALENCY = {7: 3, 8: 2, 16: 2}

# for every graph
# definedAtom = [
#     'C','N','O','S','F','Si','P','Cl','Br','Mg',
#     'Na','Ca','Fe','As','Al','I','B','V','K','Tl',
#     'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', # H?
#     'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr',
#     'Pt','Hg','Pb',
#     'Unknown'
# ] # 44
# definedAtom = ['Unknown']+pd.read_csv('./catcvae/utils/elements.csv')['Symbol'].tolist() #1+118
definedAtom = pd.read_csv('./catcvae/utils/elements_ord.csv')['Symbol'].tolist() #66 #60 #67
NUMBER_OF_ATOM = len(definedAtom)
definedBond = [
    Chem.rdchem.BondType.SINGLE, 
    Chem.rdchem.BondType.DOUBLE, 
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.DATIVE,
]
NUMBER_OF_BOND = len(definedBond)

atom_labels = [0]+list(definedAtom)
atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}

bond_labels = [Chem.rdchem.BondType.ZERO] + list(definedBond)
bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}

max_atom_number = 100
matrix_size = max_atom_number*(1 + len(atom_encoder_m) + (max_atom_number * len(bond_encoder_m)))

####################
###### GRAPH #######
####################

num_node_features = 101 #101 #94 #100 #94 #152 #154 #79 deepchem
num_edge_features = 11 #11 #10

# MoleculeGraph
class MoleculeGraph:
    def __init__(self, smiles):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.cliques = None
        self.edges = None
        self.graph_size = 0
        self.node_index = []
        self.node_features = []
        self.edge_index = []
        self.edge_features = []

    def getMoleculeGraph(self):
        return self.mol, self.smiles, self.cliques, self.edges, self.graph_size, \
            self.node_index, self.node_features, self.edge_index, self.edge_features,

    def __str__(self):
        if not isinstance(self, AtomGraph):
            return "Molecule Graph: "+self.smiles+"\n"+ \
                "Node Index:"+str(self.node_index)+"\n"+ \
                "Edge Index:"+str(self.edge_index)+"\n"+ \
                "Cliques:"+str(self.cliques)
        else:
            return "Molecule Graph: "+self.smiles+"\n"+ \
                "Node Index:"+str(self.node_index)+"\n"+ \
                "Edge Index:"+str(self.edge_index)

# ATOM-based
class AtomGraph(MoleculeGraph):
    def __init__(self, smiles, normalize=True):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.normalize =normalize
        graph_size, node_index, node_attr, edge_index, edge_attr = self.mol_to_graph(self.mol, normalize)
        self.graph_size = graph_size
        self.node_index = node_index
        self.node_features = node_attr
        self.edge_index = edge_index
        self.edge_features = edge_attr
        # for only atom graph
        self.cliques = [[i] for i in node_index]
        self.edges = edge_index

    # from deepchem
    # ref: https://github.com/deepchem/deepchem/blob/master/deepchem/feat/graph_features.py#L14
    def get_atom_features(self, atom,
                          # bool_id_feat=False,
                          explicit_H=False,
                          use_chirality=False):
        # Helper method used to compute per-atom feature vectors.
        from rdkit import Chem
        results = one_of_k_encoding_unk(atom.GetSymbol(), definedAtom+['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                [abs(atom.GetFormalCharge())] + \
                one_of_k_encoding(np.sign(atom.GetFormalCharge()), [-1, 0, 1]) + \
                [atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
                Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        # not used in this study
        if not explicit_H: 
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),['R', 'S']) \
                        + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)
    
    def get_bond_features(self, bond, use_chirality=False, use_extended_chirality=False):
        # Helper method used to compute bond feature vectors.
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This method requires RDKit to be installed.")
        bt = bond.GetBondType()
        bond_feats = [
            bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
            bt == Chem.rdchem.BondType.DATIVE,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
        if use_chirality:
            bond_feats = bond_feats + one_of_k_encoding_unk(
                str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

        if use_extended_chirality:
            stereo = one_of_k_encoding(int(bond.GetStereo()), list(range(6)), True)
            stereo = [int(feature) for feature in stereo]
            bond_feats = bond_feats + stereo
            return bond_feats

        return np.array(bond_feats)

    def getAtomFeature(self, atom):
        # number of deepchem features (78)
        # atom_features = list(np.multiply(dcfeat.graph_features.atom_features(atom, use_chirality=True), 1))
        # number of deepchem features (153-5)(151)->93
        atom_features = list(np.multiply(self.get_atom_features(atom, explicit_H=True, use_chirality=True), 1))
        # additional features (79/154/152->94)
        atom_features += [np.multiply(atom.IsInRing(), 1)]
        return atom_features

    def getBondFeature(self, bond):
        # number of deepchem features (10)
        # bond_features = list(np.multiply(dcfeat.graph_features.bond_features(bond, use_chirality=True), 1))
        bond_features = list(np.multiply(self.get_bond_features(bond, use_chirality=True), 1))
        return bond_features

    def mol_to_graph(self, mol, normalize=True):
        graph_size = mol.GetNumAtoms()
        
        node_attr = [] 
        node_index = []
        for atom in mol.GetAtoms():
            node_feature = self.getAtomFeature(atom)
            if normalize:
                assert sum(node_feature) != 0, "sum of node feature is zero."
                node_attr.append(list(node_feature)/sum(node_feature))
            else:
                node_attr.append(list(node_feature))
            node_index.append(atom.GetIdx())

        edge_attr = []
        edge_index = []
        for bond in mol.GetBonds():
            edge_feature = self.getBondFeature(bond)
            # from 1 -> 2
            if normalize and sum(edge_feature) != 0:
                assert sum(edge_feature) != 0, "sum of edge feature is zero."
                edge_attr.append(list(edge_feature)/sum(edge_feature))
            else:
                edge_attr.append(list(edge_feature))
            edge_index.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            # from 2 -> 1
            if normalize and sum(edge_feature) != 0:
                assert sum(edge_feature) != 0, "sum of edge feature is zero."
                edge_attr.append(list(edge_feature)/sum(edge_feature))
            else:
                edge_attr.append(list(edge_feature))
            edge_index.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
            
        return graph_size, node_index, node_attr, edge_index, edge_attr

###########################
###### BAG OF ATOMS #######
###########################

def bagofatoms(mol):
    boa = [0 for _ in range(len(definedAtom))]
    for a in mol.GetAtoms():
        if a.GetSymbol() in definedAtom:
            boa[definedAtom.index(a.GetSymbol())] += 1
    boa_onehot = list()
    for i in range(len(definedAtom)):
        boa_onehot.append(one_of_k_encoding(boa[i], range(max_atom_number)))
    return boa, boa_onehot

#####################
###### MATRIX #######
#####################

# get atom features
def get_node_features(mol, max_atom_number, atom_encoder_m):
    max_atom_number = max_atom_number if max_atom_number is not None else mol.GetNumAtoms()
    features = np.zeros((mol.GetNumAtoms(), len(atom_encoder_m)), dtype=np.int32)
    for i, a in enumerate(mol.GetAtoms()):
        features[i, atom_encoder_m[a.GetSymbol()]] = 1
    return np.vstack((features, np.zeros((max_atom_number - features.shape[0], features.shape[1]))))

# assign atom 0 to unknown atom
def get_annotation_matrix(mol, max_atom_number, atom_encoder_m): 
    feature = get_node_features(mol, max_atom_number, atom_encoder_m)
    feature =  np.concatenate([np.array(feature), np.zeros([max_atom_number-feature.shape[0], feature.shape[1]])], 0)
    for i in range(feature.shape[0]):
        if 1 not in feature[i]:
            feature[i, 0] = 1
    return feature

# get adjacency matrix
def get_adjacency_matrix(mol, max_atom_number, bond_encoder_m, connected=False):
    A = np.zeros(shape=(max_atom_number, max_atom_number), dtype=np.int32)
    begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
    bond_type = [bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]
    A[begin, end] = bond_type
    A[end, begin] = bond_type
    adj = A
    # check all atoms have at least one bond
    if connected:
        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)
        adj = A if (degree > 0).all() else None
    # # make upper triangle matrix BondType.ZERO
    # # not use, matrix is symmetric for this study
    # for i in range(adj.shape[0]):
    #     adj[i, 0:i] = 0
    # make matrix of shape (max_atom_number, max_atom_number*len(bond_encoder_m))
    oh_list = []
    for i in range(adj.shape[0]):
        oh = np.zeros(shape=(max_atom_number, len(bond_encoder_m)), dtype=np.int32)
        for j in range(adj.shape[1]):
            oh[j, adj[i][j]] = 1
        oh_list.append(oh)
    return np.concatenate([o for o in oh_list], 1)

# mol to matrix
def mol2matrix(mol):
    length = [[0] for i in range(max_atom_number)]
    length[int(mol.GetNumAtoms())-1] = [1]
    length = np.array(length)
    annotation_matrix = get_annotation_matrix(mol, max_atom_number, atom_encoder_m)
    adjacency_matrix = get_adjacency_matrix(mol, max_atom_number, bond_encoder_m)
    return np.concatenate([length, annotation_matrix, adjacency_matrix], 1).astype(float)

# matrix to mol
def matrix2mol(matrix_input, print_result=False, correct=True):
    matrix = matrix_input
    length = matrix[:, 0].argmax().item()+1
    annotation_matrix = matrix[:length, 1:len(atom_decoder_m)+1]
    adjacency_matrix = matrix[:length, len(atom_decoder_m)+1:len(atom_decoder_m)+1+length*len(bond_decoder_m)]

    if print_result:
        print("length: ", length)
        print("annotation_matrix: ", annotation_matrix, annotation_matrix.shape)
        print("adjacency_matrix: ", adjacency_matrix, adjacency_matrix.shape)
    mol = Chem.RWMol()
    try:
        mapping_idx = {}
        for node in range(annotation_matrix.shape[0]):
            atom = annotation_matrix[node].argmax().item()
            if atom != 0:
                new_atom_idx = mol.AddAtom(Chem.Atom(atom_decoder_m[atom]))
                mapping_idx[node] = new_atom_idx

        # Extract lower triangular part without diagonal
        adjacency_matrix = adjacency_matrix.reshape(length, length, len(bond_decoder_m))
        lower_tri_indices = np.tril_indices(length, k=-1)
        i_indices = lower_tri_indices[0]
        j_indices = lower_tri_indices[1]
        bond_vector = adjacency_matrix[i_indices, j_indices]
        node_pairs_idx = list(zip(i_indices, j_indices, bond_vector))
        
        # Sort by the maximum value of each vector using map and sorted
        if correct:
            sorted_node_pairs_idx = sorted(node_pairs_idx, key=lambda x: max(x[2]), reverse=True)
        else:
            sorted_node_pairs_idx = node_pairs_idx

        for start, end, bond in sorted_node_pairs_idx:
            if start > end:
                bond = bond.argmax().item()
                if bond != 0 and start in mapping_idx and end in mapping_idx:
                    start_atom_idx = mapping_idx[start]
                    end_atom_idx = mapping_idx[end]
                    mol.AddBond(int(start_atom_idx), int(end_atom_idx), bond_decoder_m[bond])
                    # print("add bond: ", start_atom_idx, end_atom_idx, bond_decoder_m[bond])
                    if correct:
                        # check only valency exception during molecule creation
                        # add formal charge to atom: e.g. [O+], [N+], [S+]
                        # not support [O-], [N-], [S-], [NH+] etc.
                        return_problems = check_chemistryproblems(mol)
                        if len(return_problems) == 0:
                            continue
                        else:
                            for problem in return_problems:
                                error_type = problem[0]
                                atomid = problem[1]
                                if error_type == 'AtomValenceException' and len(atomid) == 2:
                                    idx = atomid[0]
                                    v = atomid[1]
                                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                                    if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1 and mol.GetAtomWithIdx(idx).GetFormalCharge() == 0:
                                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                                        return_problems_again = check_chemistryproblems(mol)
                                        if len(return_problems_again) == 0:
                                            continue
                                        else:
                                            for problem in return_problems_again:
                                                error_type = problem[0]
                                                atomid = problem[1]
                                                if error_type == 'AtomValenceException' or error_type == 'AtomDegreeOverValenceException':
                                                    mol.GetAtomWithIdx(idx).SetFormalCharge(0)
                                                    mol.RemoveBond(int(start_atom_idx), int(end_atom_idx))
                                    else:
                                        mol.RemoveBond(int(start_atom_idx), int(end_atom_idx))
                                        # print("remove bond 1: ", start_atom_idx, end_atom_idx)
                                elif error_type == 'AtomDegreeOverValenceException':
                                    mol.RemoveBond(int(start_atom_idx), int(end_atom_idx))
                                    # print("remove bond 2: ", start_atom_idx, end_atom_idx)

    except Exception as e:
        print("Error (matrix2mol): ", e)
        mol = None
    # new function to validate mol ref: moflow
    # print('Step 01:', Chem.MolToSmiles(mol)) if mol is not None else print('None')
    if correct:
        try:
            mol = correct_mol(mol)
            # print('Step 02:', Chem.MolToSmiles(mol)) if mol is not None else print('None')
            mol = clean_nonaromatic(mol)
            # print('Step 03:', Chem.MolToSmiles(mol)) if mol is not None else print('None')
            mol = clean_carbon_unconnected(mol)
            # print('Step 04:', Chem.MolToSmiles(mol)) if mol is not None else print('None')
        except Exception as e:
            print("Error (validatemol): ", e)
            mol = None
    # print('Step 05:', Chem.MolToSmiles(mol)) if mol is not None else print('None')
    return mol

# new function to validate mol ref: moflow
def check_chemistryproblems(rwmol):
    """
    Checks all chemistry problems in a molecule
    :return: True if no problems, False otherwise
    """
    return_problems = []
    try:
        # Check degree over valence
        mol = rwmol.GetMol()
        for atom in mol.GetAtoms():
            try:
                atomid = atom.GetIdx()
                atom.UpdatePropertyCache()
                # print(atomid, 'degree', atom.GetDegree(), 'valence', atom.GetExplicitValence())
                if atom.GetDegree() > atom.GetExplicitValence():
                    sum_valence = 0
                    for bond in atom.GetBonds():
                        sum_valence += bond.GetValenceContrib(atom)
                    # print(atomid, 'sum_valence', sum_valence, 'valence', atom.GetExplicitValence())
                    if sum_valence > atom.GetExplicitValence():
                        e = f"Atom {atomid} has valence {atom.GetExplicitValence()} but degree {atom.GetDegree()}"
                        return_problems.append(('AtomDegreeOverValenceException', atomid, e))
            except Exception as e:
                if 'Explicit valence' in str(e):
                    p = e.find('#')
                    e_sub = e[p:]
                    atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
                    return_problems.append(('AtomValenceException', atomid_valence, e))
        # Check sanitization
        Chem.SanitizeMol(mol)
    except Exception as e:
        problems = Chem.DetectChemistryProblems(mol)
        if len(problems) > 0:
            for problem in problems:
                if problem.GetType() == 'AtomValenceException':
                    e = str(problem.Message())
                    p = e.find('#')
                    e_sub = e[p:]
                    atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
                    return_problems.append(('AtomValenceException', atomid_valence, e))
                elif problem.GetType() == 'AtomKekulizeException' and 'non-ring' in problem.Message():
                    e = str(problem.Message())
                    p = e.find('atom')
                    e_sub = e[p:]
                    atomid = int(re.findall(r'\d+', e_sub)[0])
                    return_problems.append(('AtomKekulizeException_NonRing', atomid, e))
                else:
                    e = str(problem.Message())
                    return_problems.append((problem.GetType(), None, e))
    # for problem in return_problems:
    #     print(problem)
    return return_problems


def correct_mol(mol):
    """
    Corrects the molecule by removing or modifying problematic bonds and atoms.
    return: corrected molecule
    """
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    while True:
        flag = False
        return_problems = check_chemistryproblems(mol)
        if len(return_problems) == 0:
            break
        else:
            for problem in return_problems:
                error_type = problem[0]
                atom_id = problem[1]
                if error_type == 'AtomValenceException':
                    flag = True
                    try:
                        assert len(atom_id) == 2
                        idx = atom_id[0]
                        v = atom_id[1]
                        queue = []
                        check_idx = 0
                        for b in mol.GetAtomWithIdx(idx).GetBonds():
                            type = int(b.GetBondType())
                            queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                            if type == 12:
                                check_idx += 1
                        queue.sort(key=lambda tup: tup[1], reverse=True)

                        if queue[-1][1] == 12:
                            # return None
                            queue.sort(key=lambda tup: tup[0], reverse=True)
                            start = queue[0][2]
                            end = queue[0][3]
                            mol.RemoveBond(start, end)
                            mol.AddBond(start, end, bond_decoder_m[1])
                        elif len(queue) > 0:
                            start = queue[check_idx][2]
                            end = queue[check_idx][3]
                            t = queue[check_idx][1] - 1
                            mol.RemoveBond(start, end)
                            if t >= 1:
                                mol.AddBond(start, end, bond_decoder_m[t])
                    except Exception as e:
                        # print(f"An error occurred in correction: {e}")
                        return None
                elif error_type == 'AtomKekulizeException_NonRing':
                    flag = True
                    try:
                        mol.GetAtomWithIdx(atom_id).SetIsAromatic(False)
                    except Exception as e:
                        # print(f"An error occurred in correction: {e}")
                        return None
                else:
                    break
        if not flag:
            break
    return mol

def clean_nonaromatic(mol):
    """
    Clean non-aromatic bonds and atoms in the molecule.
    return: cleaned molecule
    """
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC and not bond.IsInRing():
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            bond.GetBeginAtom().SetIsAromatic(False)
            bond.GetEndAtom().SetIsAromatic(False)
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            atom.SetIsAromatic(False)
    try:
        mol.UpdatePropertyCache()
    except Exception as e:
        # print('[clean_nonaromatic] UpdatePropertyCache:', e)
        pass

    count_while = 0
    while True:
        try:
            Chem.SanitizeMol(mol)
            break
        except Chem.KekulizeException as e:
            # print('[clean_nonaromatic] KekulizeException:', e)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception as e:
            # print('[clean_nonaromatic] Exception:', e)
            mol = correct_mol(mol)
        mol.UpdatePropertyCache()
        count_while += 1
        if count_while > 10:
            break

    return mol


def clean_carbon_unconnected(mol):
    """
    Remove unconnected carbon atoms from the molecule.
    return: cleaned molecule
    """
    atoms_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "C" and atom.GetDegree() == 0]
    if len(atoms_to_remove) == mol.GetNumAtoms(): # remove all atoms
        return mol
    for atom_idx in sorted(atoms_to_remove, reverse=True):
        mol.RemoveAtom(atom_idx)
    mol.UpdatePropertyCache()
    return mol


# augmentation (node order shuffling)
def augment_matrix(matrix_input, max_atom_number, atom_decoder_m, bond_decoder_m):
    matrix = matrix_input.copy()
    length = matrix[:, 0]
    annotation_matrix = matrix[:, 1:len(atom_decoder_m)+1]
    adjacency_matrix = matrix[:, len(atom_decoder_m)+1:]

    permutation = np.random.permutation(length.argmax().item()+1)
    padding = np.arange(len(permutation), max_atom_number)
    permutation = np.concatenate([permutation, padding])

    length_aug = np.expand_dims(length, axis=1)
    
    annotation_matrix_aug = annotation_matrix[permutation, :]
    annotation_matrix_aug

    adjacency_matrix_aug = adjacency_matrix.reshape(max_atom_number, max_atom_number, len(bond_decoder_m))
    # upper_tri_indices = np.triu_indices(max_atom_number, k=1)
    # adjacency_matrix_aug[upper_tri_indices] = np.transpose(adjacency_matrix_aug, axes=(1, 0, 2))[upper_tri_indices]
    adjacency_matrix_aug = adjacency_matrix_aug[permutation, :, :]
    adjacency_matrix_aug = adjacency_matrix_aug[:, permutation, :]
    # pattern = np.zeros(len(bond_decoder_m))
    # pattern[0] = 1.0
    # for i in range(max_atom_number):
    #     adjacency_matrix_aug[i, i:max_atom_number] = pattern
    adjacency_matrix_aug = adjacency_matrix_aug.reshape(max_atom_number, max_atom_number*len(bond_decoder_m))

    # print(length.shape)
    # print(annotation_matrix_aug.shape)
    # print(adjacency_matrix_aug.shape)
    matrix_aug = np.concatenate([length_aug, annotation_matrix_aug, adjacency_matrix_aug], 1).astype(float)
    return matrix_aug
