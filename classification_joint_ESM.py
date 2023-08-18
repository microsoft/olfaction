# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

## Todo: 
## Create copy of utils.predict() to include add_features
## Create custom GCNPredictor class to include add_features in MLP head
## Pass in OR_logits into main model predict
## Compare performance

import numpy as np
import torch
import torch.nn as nn

from dgllife.model import load_pretrained
from dgllife.utils import EarlyStopping, Meter, SMILESToBigraph
from torch.optim import Adam
from torch.utils.data import DataLoader
from itertools import zip_longest
import numpy as np

from utils import collate_molgraphs, load_model, predict, predict_OR_feat

def run_a_train_epoch(args, epoch, model, data_loader1, data_loader2, loss_criterion1, loss_criterion2, optimizer):
    model.train()
    train_meter1 = Meter()
    train_meter2 = Meter()
    
    # Use itertools.zip_longest to handle different-length dataloaders
    zipped_data_loaders = zip_longest(data_loader1, data_loader2)

    for batch_id, (batch_data1, batch_data2) in enumerate(zipped_data_loaders):
        if batch_data1 is not None:
            if args['cross_attention']:
                smiles, bg, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, sample_weights, seq_mask, node_mask = batch_data1
                seq_emb_arr = np.dstack(seq_embeddings)
                seq_embeddings_tensor = torch.FloatTensor(np.rollaxis(seq_emb_arr, -1)).cuda()
                seq_mask = np.vstack(seq_mask)
                seq_mask = torch.FloatTensor(seq_mask).cuda()
                
                node_mask = np.vstack(node_mask)
                node_mask = torch.FloatTensor(node_mask).cuda()
                #print(bg)
                _, logits = predict_OR_feat(args, model, bg, seq_embeddings_tensor, seq_mask, node_mask)

            else:
                smiles, bg, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, sample_weights = batch_data1
                seq_emb_arr = np.vstack(seq_embeddings)
                seq_embeddings_tensor = torch.FloatTensor(seq_emb_arr).cuda()
                _, logits = predict_OR_feat(args, model, bg, seq_embeddings_tensor)

            #print(logits)
            #print(logits.shape)
            sample_weights = torch.tensor(sample_weights).reshape(-1,1).cuda()
            #print (seq_embeddings_.shape)
            
            if len(smiles) == 1:
                # Avoid potential issues with batch normalization
                continue

            labels, masks = labels.to(args['device']), masks.to(args['device'])
            #OR_logits = predict(args, aux_model, bg)
            #print(OR_logits.shape)
            #logits = predict(args, model, bg)
            # Mask non-existing labels
            #print(data_quality.shape)
            #print((sample_weights * loss_criterion(logits, labels)).shape)
            if args['sample_weight'] == True:
                loss1 = (sample_weights * loss_criterion1(logits, labels) * (masks != 0).float()).mean()
            else:
                loss1 = (loss_criterion1(logits, labels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            train_meter1.update(logits, labels, masks)
            #print (logits[0])
            #print(labels.shape)
        if batch_data2 is not None:
            smiles2, bg2, labels2, masks2 = batch_data2
            
            if len(smiles2) == 1:
                # Avoid potential issues with batch normalization
                continue
            
            labels2, masks2 = labels2.to(args['device']), masks2.to(args['device'])
            
            scent_logits = predict(args, model, bg2)
            
            loss2 = (loss_criterion2(scent_logits, labels2) * (masks2 != 0).float()).mean()
            #print("Scent loss: ", loss2)
            #loss += loss2
            
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            train_meter2.update(scent_logits, labels2, masks2)
        if batch_id % args['print_every'] == 0:
            print('M2OR epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader1), loss1.item()))
            print('Percept epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader2), loss2.item()))

    train_score1 = np.mean(train_meter1.compute_metric(args['metric']))
    print('M2OR Dataset 1; epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score1))
    
    train_score2 = np.mean(train_meter2.compute_metric(args['metric']))
    print('GS_LF Dataset 2; epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score2))

def run_an_eval_epoch_OR(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            if args['cross_attention']:
                smiles, bg, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, sample_weights, seq_mask, node_mask = batch_data
                seq_emb_arr = np.dstack(seq_embeddings)
                seq_embeddings_tensor = torch.FloatTensor(np.rollaxis(seq_emb_arr, -1)).cuda()
                seq_mask = np.vstack(seq_mask)
                seq_mask = torch.FloatTensor(seq_mask).cuda()
                node_mask = np.vstack(node_mask)
                node_mask = torch.FloatTensor(node_mask).cuda()
                # Discard scent logits
                _, logits = predict_OR_feat(args, model, bg, seq_embeddings_tensor, seq_mask, node_mask)

            else:
                smiles, bg, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, sample_weights = batch_data
                seq_emb_arr = np.vstack(seq_embeddings)
                seq_embeddings_tensor = torch.FloatTensor(seq_emb_arr).cuda()
                # Discard scent logits
                _, logits = predict_OR_feat(args, model, bg, seq_embeddings_tensor)

            """
            smiles, bg, labels, masks, ids, seq_ids, sequences_dict, seq_embeddings, data_quality = batch_data
            seq_emb_arr = np.vstack(seq_embeddings)
            seq_embeddings_tensor = torch.FloatTensor(seq_emb_arr).cuda()
            """
            labels = labels.to(args['device'])
            #OR_logits = predict(args, aux_model, bg)
            #logits = predict_OR_feat(args, model, bg, seq_embeddings_tensor)
            #logits = predict(args, model, bg)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric']))

def run_an_eval_epoch(args, model, data_loader):
    ## Eval code for scent head
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            scent_logits = predict(args, model, bg)
            #logits = predict_OR_feat(args, model, bg, OR_logits)
            #logits = predict(args, model, bg)
            
            eval_meter.update(scent_logits, labels, masks)
            #eval_meter.update(OR_logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric']))


def main(args, exp_config, train_set, val_set, test_set):
    if args['featurizer_type'] != 'pre_train':
        print(exp_config)
        print(args['node_featurizer'].feat_size())
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
        if args['edge_featurizer'] is not None:
            exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    exp_config.update({
        'n_tasks': args['n_tasks'],
        'model': args['model']
    })
    
    exp_config['max_seq_len'] = args['max_seq_len']
    exp_config['max_node_len'] = args['max_node_len']
    exp_config['mol2prot_dim'] = args['mol2prot_dim']
    

    train_loader_OR = DataLoader(dataset=train_set[0], batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader_OR = DataLoader(dataset=val_set[0], batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader_OR = DataLoader(dataset=test_set[0], batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    train_loader_scent = DataLoader(dataset=train_set[1], batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader_scent = DataLoader(dataset=val_set[1], batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader_scent = DataLoader(dataset=test_set[1], batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    exp_config['add_feat_size'] = args['add_feat']
    if args['pretrain']:
        args['num_epochs'] = 0
        if args['featurizer_type'] == 'pre_train':
            model = load_pretrained('{}_{}'.format(
                args['model'], args['dataset'])).to(args['device'])
        else:
            model = load_pretrained('{}_{}_{}'.format(
                args['model'], args['featurizer_type'], args['dataset'])).to(args['device'])
    elif args['curriculum'] == True:
        from dgllife.model.model_zoo.mlp_predictor import MLPPredictor
        ## here we update n_tasks to be the number of tasks in the previous dataset (M2OR usually)
        exp_config.update({'n_tasks': args['prev_data_n_tasks']})
        ## Change model to GCN type to load M2OR model temporarily (super hacky)
        #exp_config.update({'model': 'GCN'})
        ## Then we initialize an empty GCN model with the same architecture as the previous model
        model = load_model(exp_config).to(args['device'])
        gnn_out_feats = model.gnn.hidden_feats[-1]
        ## plumbing issue - forward pass of GCNORPredictor results in issue loading m2or weights, so 
        ## we temporarily replace the MLP head with one that matches the shape of M2OR model
        model.predict = MLPPredictor(2 * gnn_out_feats, exp_config['predictor_hidden_feats'],
                                            exp_config['n_tasks'], dropout = exp_config['dropout'])
        ## Now we load the model weights for the previously trained model, in this case Uniprot-M2OR GCN
        checkpoint = torch.load(args['prev_model_path'] + '/model.pth', map_location=args['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        #$ Now that the model weights are loaded, lets revert n_tasks back to the current dataset's number of tasks (GS-LF)
        exp_config.update({'n_tasks': args['n_tasks']})
        #exp_config.update({'model': 'GCN_OR'})
        
        # I now initialize a new head for the model, such that the GCN is using the M2oR weights as initialization but the MLP head is new for GS-LF
        gnn_out_feats = model.gnn.hidden_feats[-1]
        ## go back and change MLP head to match size to fit OR logits and mol emb
        model.predict = MLPPredictor(2 * gnn_out_feats + exp_config['add_feat_size'], exp_config['predictor_hidden_feats'],
                                exp_config['n_tasks'], exp_config['dropout'])
        model = model.to(args['device'])
        loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                         weight_decay=exp_config['weight_decay'])
        stopper = EarlyStopping(patience=exp_config['patience'],
                                filename=args['result_path'] + '/model.pth',
                                metric=args['metric'])
    else:
        model = load_model(exp_config).to(args['device'])
        loss_criterion1 = nn.BCEWithLogitsLoss(reduction='none')
        loss_criterion2 = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                         weight_decay=exp_config['weight_decay'])
        stopper = EarlyStopping(patience=exp_config['patience'],
                                filename=args['result_path'] + '/model.pth',
                                metric=args['metric'])
    
    """
    ## here we update n_tasks to be the number of tasks in the previous dataset (M2OR usually)
    exp_config.update({'n_tasks': args['prev_data_n_tasks']})
    ## Change model to GCN type to load M2OR model temporarily (super hacky)
    exp_config.update({'model': 'GCN'})
    ## Load OR predictive model to generate OR preds as features
    OR_model = load_model(exp_config).to(args['device'])
    ## Now we load the model weights for the previously trained model, in this case Uniprot-M2OR GCN
    OR_checkpoint = torch.load(args['prev_model_path'] + '/model.pth', map_location=args['device'])
    OR_model.load_state_dict(OR_checkpoint['model_state_dict'])
    #$ Now that the model weights are loaded, lets revert n_tasks back to the current dataset's number of tasks (GS-LF)
    exp_config.update({'n_tasks': args['n_tasks']})
    exp_config.update({'model': 'GCN_OR'})
    """
    eval = args['eval']
    if not eval:
        for epoch in range(args['num_epochs']):
            # Train
            #run_a_train_epoch(args, epoch, model, OR_model, train_loader, loss_criterion, optimizer)
            run_a_train_epoch(args, epoch, model, train_loader_OR, train_loader_scent, loss_criterion1, loss_criterion2, optimizer)

            # Validation and early stop
            #val_score = run_an_eval_epoch(args, model, OR_model, val_loader)
            val_score_percept = run_an_eval_epoch(args, model, val_loader_scent)
            val_score_OR = run_an_eval_epoch_OR(args, model, val_loader_OR)
            val_score = (val_score_percept + val_score_OR)/2
            early_stop = stopper.step(val_score, model)
            print('Percept: epoch {:d}/{:d}, validation {} {:.4f}, best averaged validation {} {:.4f}'.format(
                epoch + 1, args['num_epochs'], args['metric'],
                val_score_percept, args['metric'], stopper.best_score))
            print('M2OR: epoch {:d}/{:d}, validation {} {:.4f}, best averaged validation {} {:.4f}'.format(
                epoch + 1, args['num_epochs'], args['metric'],
                val_score_OR, args['metric'], stopper.best_score))

            if early_stop:
                break

    if not args['pretrain']:
        stopper.load_checkpoint(model)
        print('loaded model')
    #val_score = run_an_eval_epoch(args, model, OR_model, val_loader)
    val_score = run_an_eval_epoch(args, model, val_loader_scent)
    test_score = run_an_eval_epoch(args, model, test_loader_scent)
    
    val_score_OR = run_an_eval_epoch_OR(args, model, val_loader_OR)
    test_score_OR = run_an_eval_epoch_OR(args, model, test_loader_OR)
    
    print('Scent val {} {:.4f}'.format(args['metric'], val_score))
    print('Scent test {} {:.4f}'.format(args['metric'], test_score))
    
    print('OR val {} {:.4f}'.format(args['metric'], val_score_OR))
    print('OR test {} {:.4f}'.format(args['metric'], test_score_OR))

    with open(args['result_path'] + '/' + args['seed'] + '_eval.txt', 'w') as f:
        f.write('Seed: {}'.format(args['seed']))
        if not args['pretrain']:
            f.write('Best average val {}: {}\n'.format(args['metric'], stopper.best_score))
        f.write('OR Val {}: {}\n'.format(args['metric'], val_score_OR))
        f.write('Percept Val {}: {}\n'.format(args['metric'], val_score))
        f.write('OR Test {}: {}\n'.format(args['metric'], test_score_OR))
        f.write('Percept Test {}: {}\n'.format(args['metric'], test_score))

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    seeds = [42, 10, 16, 24, 73]
    for seed in seeds:
        print('Seed no:' + str(seed))
        torch.manual_seed(seed)

        from utils import init_featurizer, mkdir_p, split_dataset, get_configure

        parser = ArgumentParser('Multi-label Binary Classification')
        """
        parser.add_argument('-d', '--dataset', choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                        'ToxCast', 'HIV', 'PCBA', 'Tox21'],
                            help='Dataset to use')
        """
        
        parser.add_argument('-d', '--dataset', choices=['M2OR', 'GS_LF', 'M2OR_Pairs', 'joint'], default='joint',
                            help='Dataset to use (only M2OR and GS_LF are supported)')
        
        
        parser.add_argument('-mo', '--model', choices=['GCN', 'GAT', 'GCN_OR', 'MolOR_Joint', 'MolOR', 'Weave', 'MPNN', 'AttentiveFP',
                                                    'gin_supervised_contextpred',
                                                    'gin_supervised_infomax',
                                                    'gin_supervised_edgepred',
                                                    'gin_supervised_masking',
                                                    'NF'],
                            help='Model to use')
        parser.add_argument('-f', '--featurizer-type', choices=['canonical', 'attentivefp'],
                            help='Featurization for atoms (and bonds). This is required for models '
                                'other than gin_supervised_**.')
        parser.add_argument('-pp', '--preprocess', choices = ['original', 'filtered', 'two_class', 'uniprot'], 
                            help = 'What dataset to load for M2OR only, can be original, filtered, two_class or uniprot labels.')
        parser.add_argument('-p', '--pretrain', action='store_true',
                            help='Whether to skip the training and evaluate the pre-trained model '
                                'on the test set (default: False)')
        parser.add_argument('-esm', '--esm_version', choices=['650m', '3B'], default='650m')
        parser.add_argument('-esm_rand', '--esm_random_weights', action='store_true', default = False)
        parser.add_argument('-mol2prot', '--mol2prot_dim', action='store_true', default = False, 
                            help= 'Before doing cross-attention, either map node_dim (usually 256) to prot_dim (usually 1280), \
                            or vice versa')
        parser.add_argument('-c', '--curriculum', action='store_true')
        parser.add_argument('-cross_att', '--cross_attention', action='store_true', default = False)
        parser.add_argument('-add_feat', '--add_feat', type=int, default=1280,
                            help= "For passing OR logits as features, specify n_tasks of previous dataset to correctly load saved model.")
        parser.add_argument('-prev', '--prev_data_n_tasks', type=int, default=152,
                            help= "For using pre-trained percept model, specify n_tasks of previous dataset to correctly load saved model.")
        parser.add_argument('-pmp', '--prev_model_path', type=str, default='M2OR_Uniprot_original_GCN',
                            help = 'For model to generate OR logits, specify path to trained model to correctly load model.')
        parser.add_argument('-w', '--sample_weight', action='store_true', default = False,
                                help='Whether to weigh loss for sample based on OR, molecule, data quality and label')
        ## Seeded as random_state = 42
        parser.add_argument('-s', '--split', default='random,random',
                            help='Dataset splitting method (default: scaffold)')
        parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                            help='Proportion of the dataset to use for training, validation and test, '
                                '(default: 0.8,0.1,0.1)')
        parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score'],
                            default='roc_auc_score',
                            help='Metric for evaluation (default: roc_auc_score)')
        parser.add_argument('-e', '--eval', action='store_true', default = False)
        parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                            help='Maximum number of epochs for training. '
                                'We set a large number by default as early stopping '
                                'will be performed. (default: 1000)')
        parser.add_argument('-nw', '--num-workers', type=int, default=0,
                            help='Number of processes for data loading (default: 0)')
        parser.add_argument('-pe', '--print-every', type=int, default=20,
                            help='Print the training progress every X mini-batches')
        parser.add_argument('-rp', '--result-path', type=str, default='classification_results',
                            help='Path to save training results (default: classification_results)')
        args = parser.parse_args().__dict__
        args['seed'] = seed
        if torch.cuda.is_available():
            args['device'] = torch.device('cuda:0')
        else:
            args['device'] = torch.device('cpu')

        args = init_featurizer(args)
        mkdir_p(args['result_path'])
        smiles_to_g = SMILESToBigraph(add_self_loop=True, node_featurizer=args['node_featurizer'],
                                    edge_featurizer=args['edge_featurizer'])
        exp_config = get_configure(args['model'], args['featurizer_type'], args['dataset'])

        if args['dataset'] == 'M2OR':
            from data.m2or import M2OR
            dataset = M2OR(smiles_to_graph=smiles_to_g, 
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'],
                        preprocess=args['preprocess'])
        elif args['dataset'] == 'GS_LF':
            from data.m2or import GS_LF
            dataset = GS_LF(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'M2OR_Pairs':
            from data.m2or import M2OR_Pairs
            dataset = M2OR_Pairs(smiles_to_graph=smiles_to_g, weighted_samples=args['sample_weight'],
                                cross_attention=args['cross_attention'], load_full=True, 
                                esm_random_weights=args['esm_random_weights'], esm_model=args['esm_version'],
                                n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])            
            args['max_seq_len'] = dataset.max_seq_len
            args['max_node_len'] = dataset.max_node_len
            
            """
            else:
            dataset = M2OR_Pairs(smiles_to_graph=smiles_to_g, weighted_samples=args['sample_weight'], 
                                    load_full=True, n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
            """
        
        elif args['dataset'] == 'joint':
            from data.m2or import M2OR_Pairs
            from data.m2or import GS_LF
            dataset_scent = GS_LF(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
            dataset_OR = M2OR_Pairs(smiles_to_graph=smiles_to_g, weighted_samples=args['sample_weight'],
                                cross_attention=args['cross_attention'], load_full=True, 
                                esm_random_weights=args['esm_random_weights'], esm_model=args['esm_version'],
                                n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])     
            args['n_tasks'] = [dataset_OR.n_tasks, dataset_scent.n_tasks]
            args['max_seq_len'] = dataset_OR.max_seq_len
            args['max_node_len'] = dataset_OR.max_node_len
            
        

        """
        if args['dataset'] == 'MUV':
            from dgllife.data import MUV
            dataset = MUV(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'BACE':
            from dgllife.data import BACE
            dataset = BACE(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'BBBP':
            from dgllife.data import BBBP
            dataset = BBBP(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'ClinTox':
            from dgllife.data import ClinTox
            dataset = ClinTox(smiles_to_graph=smiles_to_g,
                            n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'SIDER':
            from dgllife.data import SIDER
            dataset = SIDER(smiles_to_graph=smiles_to_g,
                            n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'ToxCast':
            from dgllife.data import ToxCast
            dataset = ToxCast(smiles_to_graph=smiles_to_g,
                            n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'HIV':
            from dgllife.data import HIV
            dataset = HIV(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'PCBA':
            from dgllife.data import PCBA
            dataset = PCBA(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'Tox21':
            from dgllife.data import Tox21
            dataset = Tox21(smiles_to_graph=smiles_to_g,
                            n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        else:
            raise ValueError('Unexpected dataset: {}'.format(args['dataset']))
        """
        #args['n_tasks'] = dataset.n_tasks
        OR_split, scent_split = map(str, args['split'].split(','))
        print('OR Split: ' + OR_split, 'Scent split: ' + scent_split)
        args['split'] = OR_split
        train_set_OR, val_set_OR, test_set_OR = split_dataset(args, dataset_OR)
        args['split'] = scent_split
        train_set_scent, val_set_scent, test_set_scent = split_dataset(args, dataset_scent)
        train_set = (train_set_OR, train_set_scent)
        val_set = (val_set_OR, val_set_scent)
        test_set = (test_set_OR, test_set_scent)
        main(args, exp_config, train_set, val_set, test_set)