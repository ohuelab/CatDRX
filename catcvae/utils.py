from rdkit import Chem
import networkx as nx

# mol with atom index
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

# smiles to mol
def mol_to_smiles(mol):
    smiles = Chem.MolToSmiles(mol)
    try:
        smiles = Chem.CanonSmiles(smiles)
    except:
        pass
    return smiles

# mol to smiles
def smiles_to_mol(smiles, with_atom_index=True, kekulize=False):
    try:
        smiles = Chem.CanonSmiles(smiles)
    except:
        pass
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            assert False, "[smiles_to_mol] Invalid SMILES: %s" % smiles
    if with_atom_index:
        mol = mol_with_atom_index(mol)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol

# convert mol to topology (atom symbol, no consider bond type)
def topology_checker(mol):
    topology = nx.Graph()
    for atom in mol.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        # Add the bonds as edges
        topology.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), type=bond.GetBondType())
        # # Add the bonds as edges
        # for bonded in atom.GetNeighbors():
        #     topology.add_edge(atom.GetIdx(), bonded.GetIdx())
    return topology

# check same atom (atom from nx)
def is_same_atom(a1, a2):
    return a1['symbol'] == a2['symbol']

# check same bond (bond from nx)
def is_same_bond(b1, b2):
    return b1['type'] == b2['type']

# check graph is isomorphic
def is_isomorphic(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2, node_match=is_same_atom, edge_match=is_same_bond)

# check graph is isomorphic only atom
def is_isomorphic_atom(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2, node_match=is_same_atom)

# one-hot encoding (only allowable set)
def one_of_k_encoding(x, allowable_set):
    # Maps inputs only in the allowable set 
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set {1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

# one-hot encoding (unknown to last element)
def one_of_k_encoding_unk(x, allowable_set):
    # Maps inputs not in the allowable set to the last element
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# one-hot encoding (unknown all zeros)
def one_of_k_encoding_none(x, allowable_set):
    # Maps inputs not in the allowable set to zero list
    if x not in allowable_set:
        x = [0 for i in range(len(allowable_set))]
    return list(map(lambda s: x == s, allowable_set))

# nodes and edges to nx
def g_to_nx(nodes, edges):
    G = nx.Graph()

    for n in nodes:
        G.add_node(n,idx=n)
    for e1, e2 in edges:
        G.add_edge(e1, e2)

    return G
