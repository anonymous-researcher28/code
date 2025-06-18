#!/usr/bin/env python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import dgl
from pubchemfp import GetPubChemFPs


class FinH2AN(nn.Module):
    def __init__(self, args):
        super(FinH2AN, self).__init__()
        self.is_classif = (args.hp_data_type == 'classification')
        self.hp_dropout = args.hp_dropout
        self.act = nn.ReLU()
        self.encoder = HFPN(args)
        self.ffn = nn.Sequential(
            nn.Dropout(self.hp_dropout),
            nn.Linear(args.hp_hidden_dim, args.hp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hp_dropout),
            nn.Linear(args.hp_hidden_dim, args.hp_output_dim)
        )
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = self.encoder(input_data)
        x = self.ffn(x)
        if self.is_classif and not self.training:
            x = self.sigmoid(x)
        return x

class MolHyGAN(nn.Module):
    def __init__(self, in_feat, e_dim, q_dim, v_dim, maccs_dim, pubchem_dim, erg_dim,
                 num_class=1, he_dim=512, extra_dropout=0.2):
        super(MolHyGAN, self).__init__()
        self.e_dim = e_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.extra_dropout = extra_dropout
        self.w1 = nn.Linear(in_feat, he_dim)
        self.w2 = nn.Linear(he_dim, e_dim)  
        self.w3 = nn.Linear(he_dim, q_dim)  
        self.w4 = nn.Linear(he_dim, v_dim)
        self.w5_maccs   = nn.Linear(maccs_dim, q_dim)
        self.w5_pubchem = nn.Linear(pubchem_dim, q_dim)
        self.w5_erg     = nn.Linear(erg_dim, q_dim)
        self.w6_maccs   = nn.Linear(q_dim, e_dim)
        self.w7_maccs   = nn.Linear(q_dim, e_dim)
        self.w6_pubchem = nn.Linear(q_dim, e_dim)
        self.w7_pubchem = nn.Linear(q_dim, e_dim)
        self.w6_erg     = nn.Linear(q_dim, e_dim)
        self.w7_erg     = nn.Linear(q_dim, e_dim)
        self.extra_mlp = nn.Sequential(
            nn.Linear(3 * e_dim, 2 * e_dim),
            nn.ReLU(),
            nn.Dropout(self.extra_dropout),
            nn.Linear(2 * e_dim, e_dim),
            nn.ReLU()  
        )
        self.cls = nn.Linear(e_dim, num_class)

    def hyperedge_to_node(self, g, first_layer=True):
        if first_layer:
            e_feat = self.w1(g.nodes['hyperedge'].data['feat'])
            g.nodes['hyperedge'].data['feat_trans'] = e_feat
            g.nodes['hyperedge'].data['k'] = self.w3(e_feat)
            g.nodes['hyperedge'].data['v'] = self.w4(e_feat)
        for ntype, proj in zip(['maccs', 'pubchem', 'erg'],
                               [self.w5_maccs, self.w5_pubchem, self.w5_erg]):
            g.nodes[ntype].data['q'] = proj(g.nodes[ntype].data['feat'])
        for ntype in ['maccs', 'pubchem', 'erg']:
            def message_func(edges):
                att = (edges.src['k'] * edges.dst['q']).sum(dim=-1) / math.sqrt(self.q_dim)
                att = F.leaky_relu(att)
                return {'att': att, 'v': edges.src['v']}
            def reduce_func(nodes):
                alpha = F.softmax(nodes.mailbox['att'], dim=1)
                return {'feat_new': torch.sum(alpha.unsqueeze(-1) * nodes.mailbox['v'], dim=1)}
            g.update_all(message_func, reduce_func, etype=('hyperedge', 'con', ntype))
            g.nodes[ntype].data['feat'] = g.nodes[ntype].data['feat_new']
            #IF FREESOLV: UNCOMMENT THE FOLLOWING
            # if 'feat_new' in g.nodes[ntype].data:
            #    g.nodes[ntype].data['feat'] = g.nodes[ntype].data['feat_new']
        return g

    def node_to_hyperedge(self, g):
        messages = []
        for ntype in ['maccs', 'pubchem', 'erg']:
            if ntype == 'maccs':
                trans_k, trans_v = self.w6_maccs, self.w7_maccs
            elif ntype == 'pubchem':
                trans_k, trans_v = self.w6_pubchem, self.w7_pubchem
            else:
                trans_k, trans_v = self.w6_erg, self.w7_erg
            node_feat = g.nodes[ntype].data['q']
            k_to_he = trans_k(node_feat)
            v_to_he = trans_v(node_feat)
            g.nodes[ntype].data['k_to_he'] = k_to_he
            g.nodes[ntype].data['v_to_he'] = v_to_he
            he_feat = g.nodes['hyperedge'].data['feat_trans']
            g.nodes['hyperedge'].data['q_from_nodes'] = self.w2(he_feat)
            def message_func(edges):
                att = (edges.src['k_to_he'] * edges.dst['q_from_nodes']).sum(dim=-1) / math.sqrt(self.q_dim)
                att = F.leaky_relu(att)
                return {'att': att, 'v': edges.src['v_to_he']}
            def reduce_func(nodes):
                alpha = F.softmax(nodes.mailbox['att'], dim=1)
                return {'msg': torch.sum(alpha.unsqueeze(-1) * nodes.mailbox['v'], dim=1)}
            g.update_all(message_func, reduce_func, etype=(ntype, 'in', 'hyperedge'))
            messages.append(g.nodes['hyperedge'].data.pop('msg'))
            #IF FREESOLV: UNCOMMENT THE FOLLOWING
            # g.update_all(message_func, reduce_func, etype=(ntype, 'in', 'hyperedge'))
            # if 'msg' in g.nodes['hyperedge'].data:
            #     messages.append(g.nodes['hyperedge'].data.pop('msg'))
            # else:
            #     n_he = g.number_of_nodes('hyperedge')
            #     dev = g.nodes['hyperedge'].data['feat_trans'].device
            #     messages.append(torch.zeros(n_he, self.e_dim, device=dev))

        he_feat_new = torch.cat(messages, dim=-1)
        g.nodes['hyperedge'].data['feat_updated'] = he_feat_new
        return he_feat_new

    def forward(self, g, first_layer=True, last_layer=False):
        with g.local_scope():
            g = self.hyperedge_to_node(g, first_layer=first_layer)
            he_feat_new = self.node_to_hyperedge(g)
            x = self.extra_mlp(he_feat_new)
            if last_layer:
                return self.cls(x)
            else:
                x = F.dropout(x, p=self.extra_dropout, training=self.training)
                return x


class HFPN(nn.Module):
    def __init__(self, args):
        super(HFPN, self).__init__()
        self.dropout_rate = args.hp_dropout
        self.device = torch.device("cuda" if args.hp_cuda else "cpu")
        self.hidden_dim = args.hp_hidden_dim
        self.he_dim = args.he_dim
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)

        self.maccs_length = 167   
        self.erg_length = 441    
        self.pubchem_length = 881
        self.fp_hyper_dim = self.maccs_length + self.erg_length + self.pubchem_length

        self.molhygan = MolHyGAN(
            in_feat=self.fp_hyper_dim,
            e_dim=getattr(args, 'hp_mol_e_dim', 64),
            q_dim=getattr(args, 'hp_mol_q_dim', 64),
            v_dim=getattr(args, 'hp_mol_v_dim', 64),
            maccs_dim=self.maccs_length,
            pubchem_dim=self.pubchem_length,
            erg_dim=self.erg_length,
            he_dim=self.he_dim,
            extra_dropout=0,
            num_class=1
        )
        self.residual_proj = nn.Linear(self.fp_hyper_dim, 128)
        self.fc1 = nn.Linear(getattr(args, 'hp_mol_e_dim', 64), getattr(args, 'hp_fp_proj_dim', 256))
        self.fc2 = nn.Linear(getattr(args, 'hp_fp_proj_dim', 256), self.hidden_dim)

    def convert_fp(self, fp, expected_length):
        if isinstance(fp, np.ndarray):
            if fp.shape[0] == expected_length:
                return fp
            else:
                raise ValueError("Fingerprint array has unexpected size.")
        arr = np.zeros((expected_length,), dtype=int)
        if hasattr(fp, "GetNonzeroElements"):
            elems = fp.GetNonzeroElements()
            for bit, _ in elems.items():
                if bit < expected_length:
                    arr[bit] = 1
        else:
            DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def forward(self, smiles):
        list_maccs, list_erg, list_pubchem = [], [], []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            arr_maccs = self.convert_fp(fp_maccs, self.maccs_length)
            fp_erg = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
            arr_erg = self.convert_fp(fp_erg, self.erg_length)
            fp_pubchem = GetPubChemFPs(mol)
            arr_pubchem = self.convert_fp(fp_pubchem, self.pubchem_length)
            list_maccs.append(arr_maccs)
            list_erg.append(arr_erg)
            list_pubchem.append(arr_pubchem)
        
        batch_size = len(list_maccs)
        def get_edges(fp_list, total_bits):
            src, dst = [], []
            for i, arr in enumerate(fp_list):
                active = np.where(arr == 1)[0]
                src.extend(active.tolist())
                dst.extend([i] * len(active))
            return np.array(src), np.array(dst)
        
        src_maccs, dst_maccs = get_edges(list_maccs, self.maccs_length)
        src_pubchem, dst_pubchem = get_edges(list_pubchem, self.pubchem_length)
        src_erg, dst_erg = get_edges(list_erg, self.erg_length)
        data_dict = {
            ('maccs', 'in', 'hyperedge'): (src_maccs, dst_maccs),
            ('pubchem', 'in', 'hyperedge'): (src_pubchem, dst_pubchem),
            ('erg', 'in', 'hyperedge'): (src_erg, dst_erg),
            ('hyperedge', 'con', 'maccs'): (dst_maccs, src_maccs),
            ('hyperedge', 'con', 'pubchem'): (dst_pubchem, src_pubchem),
            ('hyperedge', 'con', 'erg'): (dst_erg, src_erg),
        }
        num_nodes_dict = {
            'maccs': self.maccs_length,
            'pubchem': self.pubchem_length,
            'erg': self.erg_length,
            'hyperedge': batch_size
        }
        hyG = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict).to(self.device)
        hyG.nodes['maccs'].data['feat'] = torch.eye(self.maccs_length, device=self.device)
        hyG.nodes['pubchem'].data['feat'] = torch.eye(self.pubchem_length, device=self.device)
        hyG.nodes['erg'].data['feat'] = torch.eye(self.erg_length, device=self.device)
        hyper_feats = np.concatenate([np.array(list_maccs),
                                      np.array(list_pubchem),
                                      np.array(list_erg)], axis=1)
        hyG.nodes['hyperedge'].data['feat'] = torch.tensor(hyper_feats, dtype=torch.float32, device=self.device)
        fp_embed = self.molhygan(hyG, first_layer=True, last_layer=False)
        hyper_feats_tensor = torch.tensor(hyper_feats, dtype=torch.float32, device=self.device)
        fused = fp_embed + self.residual_proj(hyper_feats_tensor)
        x = self.fc1(fused)
        x = self.dropout(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


