# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import errno
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

def init_featurizer(args):
    """Initialize node/edge featurizer

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with featurizers updated
    """
    if args['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                         'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
        args['featurizer_type'] = 'pre_train'
        args['node_featurizer'] = PretrainAtomFeaturizer()
        args['edge_featurizer'] = PretrainBondFeaturizer()
        return args

    if args['featurizer_type'] == 'canonical':
        from dgllife.utils import CanonicalAtomFeaturizer
        args['node_featurizer'] = CanonicalAtomFeaturizer()
    elif args['featurizer_type'] == 'attentivefp':
        from dgllife.utils import AttentiveFPAtomFeaturizer
        args['node_featurizer'] = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect featurizer_type to be in ['canonical', 'attentivefp'], "
            "got {}".format(args['featurizer_type']))

    if args['model'] in ['Weave', 'MPNN', 'AttentiveFP']:
        if args['featurizer_type'] == 'canonical':
            from dgllife.utils import CanonicalBondFeaturizer
            args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
        elif args['featurizer_type'] == 'attentivefp':
            from dgllife.utils import AttentiveFPBondFeaturizer
            args['edge_featurizer'] = AttentiveFPBondFeaturizer(self_loop=True)
    else:
        args['edge_featurizer'] = None

    return args

def mkdir_p(path):
    """Create a folder for the given path.

    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def split_dataset(args, dataset):
    """Split the dataset

    Parameters
    ----------
    args : dict
        Settings
    dataset
        Dataset instance

    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    from dgllife.utils import ScaffoldSplitter, RandomSplitter
    train_ratio, val_ratio, test_ratio = map(float, args['split_ratio'].split(','))
    if args['split'] == 'scaffold':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args['split'] == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio, random_state = 42)
    elif args['split'] == 'iterative_stratification':
        print('Using iterative stratification')
        from skmultilearn.model_selection import IterativeStratification
        from dgl.data.utils import Subset
        import numpy as np
        np.random.RandomState(42)
        stratifier = IterativeStratification(n_splits= 2, order = 2, sample_distribution_per_fold=[0.2, 0.8])
        train_indices, orig_test_indices = next(stratifier.split(dataset.smiles, dataset.labels))
        test_smiles = [dataset.smiles[i] for i in orig_test_indices]        ##Subset(dataset, indices[offset - length : offset])
        stratifier = IterativeStratification(n_splits= 2, order = 2, sample_distribution_per_fold=[0.5, 0.5])
        rel_val_indices, rel_test_indices = next(stratifier.split(test_smiles, dataset.labels[orig_test_indices]))
        val_indices, test_indices = orig_test_indices[rel_val_indices], orig_test_indices[rel_test_indices]        
        train_set, val_set, test_set = Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args['split']))

    return train_set, val_set, test_set

def get_configure(model, featurizer_type, dataset):
    """Query for configuration

    Parameters
    ----------
    model : str
        Model type
    featurizer_type : str
        The featurization performed
    dataset : str
        Dataset for modeling

    Returns
    -------
    dict
        Returns the manually specified configuration
    """

    if featurizer_type == 'pre_train':
        with open('data/configures/{}/{}.json'.format(dataset, model), 'r') as f:
            config = json.load(f)
    else:
        ## Joint ROOT_DIR with file_path
        file_path = 'data/configures/{}/{}_{}.json'.format(dataset, model, featurizer_type)
        file_path = os.path.join(ROOT_DIR, file_path)
        print(file_path)
        if not os.path.isfile(file_path):
            return NotImplementedError('Model {} on dataset {} with featurization {} has not been '
                                       'supported'.format(model, dataset, featurizer_type))
        with open(file_path, 'r') as f:
            config = json.load(f)
        print(config)
    return config

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    ids, seq_ids, sequences_dict, seq_embeddings = None, None, None, None
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    elif len(data[0]) == 6:
        smiles, graphs, labels, masks, ids, node_mask = map(list, zip(*data))
    elif len(data[0]) == 7:
        idxs, smiles, graphs, labels, masks, ids, node_mask = map(list, zip(*data))
    elif len(data[0]) == 9:
        smiles, graphs, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, sample_weights = map(list, zip(*data))
    elif len(data[0]) == 11:
        smiles, graphs, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, sample_weights, seq_mask, node_mask = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    
    if len(data[0]) == 9:
        seq_emb_arr = np.dstack(seq_embeddings)
        seq_embeddings = torch.FloatTensor(np.rollaxis(seq_emb_arr, -1))#.cuda()
        sample_weights = torch.tensor(sample_weights).reshape(-1,1)
        return smiles, bg, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, sample_weights
    elif len(data[0]) == 6:        
        node_mask = np.vstack(node_mask)
        node_mask = torch.FloatTensor(node_mask)#.cuda()
        return smiles, bg, labels, masks, ids, node_mask
    elif len(data[0]) == 7:
        node_mask = np.vstack(node_mask)
        node_mask = torch.FloatTensor(node_mask)
        return idxs, smiles, bg, labels, masks, ids, node_mask
    elif len(data[0]) == 11:
        seq_emb_arr = np.dstack(seq_embeddings)
        seq_embeddings = torch.FloatTensor(np.rollaxis(seq_emb_arr, -1))#.cuda()
        seq_mask = np.vstack(seq_mask)
        seq_mask = torch.FloatTensor(seq_mask)#.cuda()
        
        node_mask = np.vstack(node_mask)
        node_mask = torch.FloatTensor(node_mask)#.cuda()
        sample_weights = torch.tensor(sample_weights).reshape(-1,1)

        return smiles, bg, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, sample_weights, seq_mask, node_mask
    return smiles, bg, labels, masks

def load_model(exp_configure):
    if exp_configure['model'] == 'GCN':
        from dgllife.model import GCNPredictor
        model = GCNPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    elif exp_configure['model'] == 'GCN_OR':
        from gcn_or_predictor import GCNORPredictor
        model = GCNORPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            add_feats=exp_configure['add_feat_size'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    elif exp_configure['model'] == 'GCN_joint':
        from gcn_or_predictor import GCNJointPredictor
        model = GCNJointPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            add_feats=exp_configure['pass_add_feat'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    elif exp_configure['model'] == 'MolOR': ## cross attention model for OR prediction
        from gcn_or_predictor import MolORPredictor
        model = MolORPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            add_feats=exp_configure['add_feat_size'],
            prot_feats=exp_configure['add_feat_size'],
            gnn_attended_feats=exp_configure['gnn_attended_feats'], # set to same as protein emb (1280) to do predictions on mean-aggr attended embeddings.
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            mol2_prot=exp_configure['mol2prot_dim'],
            max_seq_len=exp_configure['max_seq_len'],
            max_node_len=exp_configure['max_node_len'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    elif exp_configure['model'] == 'MolOR_Joint':
        from gcn_or_predictor import Mol_JointPredictor
        model = Mol_JointPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            add_feats=exp_configure['add_feat_size'],
            prot_feats=exp_configure['add_feat_size'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            mol2_prot=exp_configure['mol2prot_dim'],
            max_seq_len=exp_configure['max_seq_len'],
            max_node_len=exp_configure['max_node_len'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    elif exp_configure['model'] == 'GAT':
        from dgllife.model import GATPredictor
        model = GATPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            num_heads=[exp_configure['num_heads']] * exp_configure['num_gnn_layers'],
            feat_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            attn_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            alphas=[exp_configure['alpha']] * exp_configure['num_gnn_layers'],
            residuals=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'Weave':
        from dgllife.model import WeavePredictor
        model = WeavePredictor(
            node_in_feats=exp_configure['in_node_feats'],
            edge_in_feats=exp_configure['in_edge_feats'],
            num_gnn_layers=exp_configure['num_gnn_layers'],
            gnn_hidden_feats=exp_configure['gnn_hidden_feats'],
            graph_feats=exp_configure['graph_feats'],
            gaussian_expand=exp_configure['gaussian_expand'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'MPNN':
        from dgllife.model import MPNNPredictor
        model = MPNNPredictor(
            node_in_feats=exp_configure['in_node_feats'],
            edge_in_feats=exp_configure['in_edge_feats'],
            node_out_feats=exp_configure['node_out_feats'],
            edge_hidden_feats=exp_configure['edge_hidden_feats'],
            num_step_message_passing=exp_configure['num_step_message_passing'],
            num_step_set2set=exp_configure['num_step_set2set'],
            num_layer_set2set=exp_configure['num_layer_set2set'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'AttentiveFP':
        from dgllife.model import AttentiveFPPredictor
        model = AttentiveFPPredictor(
            node_feat_size=exp_configure['in_node_feats'],
            edge_feat_size=exp_configure['in_edge_feats'],
            num_layers=exp_configure['num_layers'],
            num_timesteps=exp_configure['num_timesteps'],
            graph_feat_size=exp_configure['graph_feat_size'],
            dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                                    'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.model import GINPredictor
        from dgllife.model import load_pretrained
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=exp_configure['jk'],
            dropout=0.5,
            readout=exp_configure['readout'],
            n_tasks=exp_configure['n_tasks']
        )
        model.gnn = load_pretrained(exp_configure['model'])
        model.gnn.JK = exp_configure['jk']
    elif exp_configure['model'] == 'NF':
        from dgllife.model import NFPredictor
        model = NFPredictor(
            in_feats=exp_configure['in_node_feats'],
            n_tasks=exp_configure['n_tasks'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_size=exp_configure['predictor_hidden_feats'],
            predictor_batchnorm=exp_configure['batchnorm'],
            predictor_dropout=exp_configure['dropout']
        )
    else:
        return ValueError("Expect model to be from ['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP', "
                          "'gin_supervised_contextpred', 'gin_supervised_infomax', "
                          "'gin_supervised_edgepred', 'gin_supervised_masking'], 'NF'"
                          "got {}".format(exp_configure['model']))

    return model

def predict(args, model, bg):
    bg = bg.to(args['device'])
    if args['edge_featurizer'] is None:
        node_feats = bg.ndata.pop('h').to(args['device'])
        return model(bg, node_feats)
    elif args['featurizer_type'] == 'pre_train':
        node_feats = [
            bg.ndata.pop('atomic_number').to(args['device']),
            bg.ndata.pop('chirality_type').to(args['device'])
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(args['device']),
            bg.edata.pop('bond_direction_type').to(args['device'])
        ]
        return model(bg, node_feats, edge_feats)
    else:
        node_feats = bg.ndata.pop('h').to(args['device'])
        edge_feats = bg.edata.pop('e').to(args['device'])
        return model(bg, node_feats, edge_feats)
    
def predict_OR_feat(args, model, bg, add_feat = None, seq_mask = None, node_mask = None):
    bg = bg.to(args['device'])
    if args['edge_featurizer'] is None:
        node_feats = bg.ndata.pop('h').to(args['device'])
        if add_feat is not None:
            #print(node_feats) - good here
            if seq_mask is None and node_mask is None: ## OR logits or ESM fixed-vector emb
                return model(bg, node_feats, add_feat)
            else: ## cross-attention forward pass
                if args['model'] == "MolOR":
                    return model(bg, node_feats, add_feat, seq_mask, node_mask, args['device'])
                else:
                    return model(bg, node_feats, add_feat, seq_mask, node_mask)
        return model(bg, node_feats)
    elif args['featurizer_type'] == 'pre_train':
        node_feats = [
            bg.ndata.pop('atomic_number').to(args['device']),
            bg.ndata.pop('chirality_type').to(args['device'])
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(args['device']),
            bg.edata.pop('bond_direction_type').to(args['device'])
        ]
        return model(bg, node_feats, edge_feats)
    else:
        node_feats = bg.ndata.pop('h').to(args['device'])
        edge_feats = bg.edata.pop('e').to(args['device'])
        return model(bg, node_feats, edge_feats)
