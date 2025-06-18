#!/usr/bin/env python
""" 
molecule processing, graph generation, dataset splitting,
"""
import random
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class MoleculeRecord:
    def __init__(self, line, config):
        self.config = config
        self.smiles = line[0]
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.targets = [float(x) if x != '' else None for x in line[1:]]
        
    def num_tasks(self):
        return len(self.targets)
    
    def update_targets(self, new_targets):
        self.targets = new_targets

class MoleculeDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.config = records[0].config if records else None
        self.scaler = None
        
    def get_smiles(self):
        return [rec.smiles for rec in self.records]
    
    def get_mols(self):
        return [rec.mol for rec in self.records]
    
    def get_targets(self):
        return [rec.targets for rec in self.records]
    
    def num_tasks(self):
        return self.records[0].num_tasks() if self.records else None
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        return self.records[idx]
    
    def shuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.records)
        
    def update_all_targets(self, targets_list):
        assert len(self.records) == len(targets_list)
        for rec, t in zip(self.records, targets_list):
            rec.update_targets(t)



ATOM_TYPE_MAX = 100
ATOM_FEATURE_DIM = 133
ATOM_FEATURE_CONFIG = {
    'atomic_number': list(range(ATOM_TYPE_MAX)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chirality': [0, 1, 2, 3],
    'num_hydrogens': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

SMILES_GRAPH_CACHE = {}

def get_atom_feature_dim():
    return ATOM_FEATURE_DIM

def onehot_encode(item, choices):
    encoding = [0] * (len(choices) + 1)
    try:
        index = choices.index(item)
    except ValueError:
        index = -1
    encoding[index] = 1
    return encoding

def compute_atom_features(atom):
    feats = onehot_encode(atom.GetAtomicNum() - 1, ATOM_FEATURE_CONFIG['atomic_number']) + \
            onehot_encode(atom.GetTotalDegree(), ATOM_FEATURE_CONFIG['degree']) + \
            onehot_encode(atom.GetFormalCharge(), ATOM_FEATURE_CONFIG['formal_charge']) + \
            onehot_encode(int(atom.GetChiralTag()), ATOM_FEATURE_CONFIG['chirality']) + \
            onehot_encode(int(atom.GetTotalNumHs()), ATOM_FEATURE_CONFIG['num_hydrogens']) + \
            onehot_encode(int(atom.GetHybridization()), ATOM_FEATURE_CONFIG['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]
    return feats

class MoleculeGraph:
    def __init__(self, smiles, config):
        self.smiles = smiles
        self.atom_features = []
        mol = Chem.MolFromSmiles(smiles)
        self.num_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            self.atom_features.append(compute_atom_features(atom))
        self.atom_features = self.atom_features[:self.num_atoms]

class BatchGraph:
    def __init__(self, smiles_list, config):
        graphs = []
        self.smiles = []
        for smi in smiles_list:
            if smi in SMILES_GRAPH_CACHE:
                graph = SMILES_GRAPH_CACHE[smi]
            else:
                graph = MoleculeGraph(smi, config)
                SMILES_GRAPH_CACHE[smi] = graph
            graphs.append(graph)
            self.smiles.append(smi)
        self.total_atoms = 1  # Start with one dummy atom
        self.atom_indices = []
        combined_features = [[0] * get_atom_feature_dim()]
        for graph in graphs:
            combined_features.extend(graph.atom_features)
            self.atom_indices.append((self.total_atoms, graph.num_atoms))
            self.total_atoms += graph.num_atoms
        self.atom_features_tensor = torch.FloatTensor(combined_features)
        
    def get_features(self):
        return self.atom_features_tensor, self.atom_indices

def build_graph_batch(smiles_list, config):
    return BatchGraph(smiles_list, config)

# -----------------------------------------
# Data Splitting
# -----------------------------------------

def extract_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

def split_by_scaffold(dataset, ratios, seed, logger):
    n_total = len(dataset)
    train_size = int(ratios[0] * n_total)
    val_size = int(ratios[1] * n_total)
    
    scaffold_to_idx = {}
    for idx, rec in enumerate(dataset.records):
        scaf = extract_scaffold(rec.smiles)
        scaffold_to_idx.setdefault(scaf, set()).add(idx)
        
    scaffold_sets = list(scaffold_to_idx.values())
    big_sets = []
    small_sets = []
    for idx_set in scaffold_sets:
        if len(idx_set) > val_size / 2 or len(idx_set) > (ratios[2] * n_total) / 2:
            big_sets.append(idx_set)
        else:
            small_sets.append(idx_set)
    random.seed(seed)
    random.shuffle(big_sets)
    random.shuffle(small_sets)
    combined_sets = big_sets + small_sets

    train_idx, val_idx, test_idx = [], [], []
    for idx_set in combined_sets:
        if len(train_idx) + len(idx_set) <= train_size:
            train_idx += list(idx_set)
        elif len(val_idx) + len(idx_set) <= val_size:
            val_idx += list(idx_set)
        else:
            test_idx += list(idx_set)
    logger.debug(f"Scaffold split: {len(scaffold_to_idx)} scaffolds; train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    train_set = MoleculeDataset([dataset.records[i] for i in train_idx])
    val_set = MoleculeDataset([dataset.records[i] for i in val_idx])
    test_set = MoleculeDataset([dataset.records[i] for i in test_idx])
    return train_set, val_set, test_set

def random_split(dataset, ratios, seed):
    dataset.shuffle(seed)
    n_total = len(dataset)
    train_end = int(ratios[0] * n_total)
    val_end = train_end + int(ratios[1] * n_total)
    train_set = MoleculeDataset(dataset.records[:train_end])
    val_set = MoleculeDataset(dataset.records[train_end:val_end])
    test_set = MoleculeDataset(dataset.records[val_end:])
    return train_set, val_set, test_set

def split_dataset(dataset, split_type, ratios, seed, logger):
    if split_type == 'random':
        return random_split(dataset, ratios, seed)
    elif split_type == 'scaffold':
        return split_by_scaffold(dataset, ratios, seed, logger)
    else:
        raise ValueError("Unknown split type.")
