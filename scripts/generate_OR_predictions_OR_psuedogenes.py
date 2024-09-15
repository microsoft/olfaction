# -*- coding: utf-8 -*-
#


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

from utils import collate_molgraphs, load_model, predict, predict_OR_feat

def run_a_train_epoch(args, epoch, model, OR_logits, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        idxs, smiles, bg, labels, masks, ids, node_masks = batch_data
        #print('seq embed shape')
        #print(seq_embeddings.shape)
        #seq_masks = seq_masks.cuda()
        #node_masks = node_masks.cuda()

        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(args['device']), masks.to(args['device'])
        ## Is it faster to iterate through each mol and generate logits for all 100 sequences by copying
        ## over graph features vs what we do now? Unlikely since torch.repeat is faster.
        #OR_logits = torch.zeros((len(smiles), seq_embeddings.shape[0])).cuda()
        #for i in range(seq_embeddings.shape[0]):
        #    ## Get i-th sequence embedding
        #    seq_embed = seq_embeddings[i]
        #    #print('seq embed shape out of dataloader')
            #print(seq_embed.shape)
        #   seq_mask = seq_masks[i]
        #    # Copy len(smiles) times into tensor of shape (len(smiles), seq_embed.shape)
        #    seq_embed = seq_embed.repeat(len(smiles), 1, 1)
        #    seq_mask = seq_mask.repeat(len(smiles), 1)
        #    #print(predict_OR_feat(args, aux_model, bg, seq_embed, seq_mask, node_masks).shape)
        #    OR_logits[:, i] = predict_OR_feat(args, aux_model, bg, seq_embed, seq_mask, node_masks).squeeze(dim=1)
        #print(OR_logits.shape) ## Should be (batch_size, 100)
        #print(OR_logits[batch_id, :len(smiles), :].shape)
        logits = predict_OR_feat(args, model, bg, OR_logits[idxs, :])
        #logits = predict(args, model, bg)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
        #print (logits[0])
        #print(labels.shape)
        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
    train_score = np.mean(train_meter.compute_metric(args['metric']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score))

def run_an_eval_epoch(args, model, OR_logits, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            idxs, smiles, bg, labels, masks, ids, node_masks = batch_data
            labels = labels.to(args['device'])
            #OR_logits = torch.zeros((len(smiles), 100)).cuda()
            #seq_masks = seq_masks.cuda()
            #node_masks = node_masks.cuda()
            #for i in range(seq_embeddings.shape[0]):
            #    ## Get i-th sequence embedding
            #    seq_embed = seq_embeddings[i]
            #    seq_mask = seq_masks[i]
                # Copy len(smiles) times into tensor of shape (len(smiles), seq_embed.shape)
            #    seq_embed = seq_embed.repeat(len(smiles), 1, 1)
            #    seq_mask = seq_mask.repeat(len(smiles), 1)
            #    OR_logits[:, i] = predict_OR_feat(args, aux_model, bg, seq_embed, seq_mask, node_masks).squeeze(dim=1)
            #print(OR_logits.shape) ## Should be (batch_size, 100)
            #print(OR_logits.shape)
            #print(OR_logits[batch_id, :len(smiles), :].shape)
            logits = predict_OR_feat(args, model, bg, OR_logits[idxs, :])
            #logits = predict_OR_feat(args, model, bg, OR_logits)
            #logits = predict(args, model, bg)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric']))

def main(args, exp_config, dataset, train_set, val_set, test_set):
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

    data_loader = DataLoader(dataset=dataset, batch_size=exp_config['batch_size'], shuffle = False,
                                collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    exp_config['add_feat_size'] = args['num_OR_logits'] ## non-OR model takes in 100 extra features for each OR seq
    exp_config['mol2prot_dim'] = args['mol2prot_dim']
    #exp_config['max_seq_len'] = args['max_seq_len']
    exp_config['max_node_len'] = args['max_node_len']  

    ## generate seq_mask, seq_embeddings
    if args['num_OR_logits'] > 0:
        import pandas as pd
        mol_OR = pd.read_csv('data/datasets/genes.csv', delimiter='\t')
        top_seqs = mol_OR['Conceptual Sequence'].value_counts().keys().tolist()
        top_seqs = [seq.replace('\\', '') for seq in top_seqs] ## get rid of gaps for embedding
        top_seqs = [seq.replace('/', '') for seq in top_seqs] ## get rid of gaps for embedding
        top_seqs = [seq.replace('*', '') for seq in top_seqs] ## get rid of gaps for embedding
        top_seqs = [seq.replace('trand=+', '') for seq in top_seqs] ## get rid of gaps for embedding
        for i, seq in enumerate(top_seqs):
            if len(seq.split("=")) != 1:
                top_seqs[i] = seq.split("|")[1]
        
        max_seq_len = len(max(top_seqs, key=len))        
        seq_masks = torch.zeros((len(top_seqs), max_seq_len))
        print(seq_masks.shape)
        for i in range(len(top_seqs)):
            seq_masks[i, :len(top_seqs[i])] = 1
            top_seqs[i] += "<pad>"*(max_seq_len - len(top_seqs[i]))
        exp_config['max_seq_len'] = max_seq_len
        from data.m2or import esm_embed
        seq_embeddings = esm_embed(top_seqs, per_residue=True, random_weights=False, esm_model_version = '650m') ## output shape: (100, max_seq_len, embedding_dim)
        print(len(seq_embeddings))#.shape)
        seq_emb_arr = np.dstack(seq_embeddings)
        seq_embeddings = torch.FloatTensor(np.rollaxis(seq_emb_arr, -1))#.cuda()
        print(seq_embeddings.shape)
        #seq_embeddings = seq_embeddings.to(args['device'])
    else:
        exp_config['max_seq_len'] = 0

    if args['pretrain']:
        args['num_epochs'] = 0
        if args['featurizer_type'] == 'pre_train':
            model = load_pretrained('{}_{}'.format(
                args['model'], args['dataset'])).to(args['device'])
        else:
            model = load_pretrained('{}_{}_{}'.format(
                args['model'], args['featurizer_type'], args['dataset'])).to(args['device'])
    elif args['curriculum'] == True:
        ## Use pre-trained GNN encoder from M2OR to initialize GCN model
        model = load_model(exp_config).to(args['device'])
        loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                         weight_decay=exp_config['weight_decay'])
        stopper = EarlyStopping(patience=exp_config['patience'],
                                filename=args['result_path'] + '/model.pth',
                                metric=args['metric'])
        exp_config['gnn_attended_feats'] = args['gnn_attended_feats']
        ## here we update n_tasks to be the number of tasks in the previous dataset (M2OR usually)
        exp_config.update({'n_tasks': args['prev_data_n_tasks']})
        ## Change model to GCN type to load M2OR model temporarily (super hacky)
        exp_config.update({'model': 'MolOR'})
        exp_config.update({'add_feat_size' : args['prot_dim']}) #args['prot_dim'] NOTE: NEED LESS HACKY WAY
        ## Load OR predictive model to generate OR preds as features
        exp_config['num_gnn_layers'] = 2 ## MolOR is 2 GNN layers
        exp_config['predictor_hidden_feats'] = 128 ## MolOR is 256 hidden feats
        exp_config['gnn_hidden_feats'] = 256 ## MolOR is 256 hidden feats
        OR_model = load_model(exp_config).to(args['device'])
        ## Now we load the model weights for the previously trained model, in this case Uniprot-M2OR GCN
        OR_checkpoint = torch.load(args['prev_model_path'] + '/model.pth', map_location=args['device'])
        OR_model.load_state_dict(OR_checkpoint['model_state_dict'])     
        ##NOTE: Trying something where in addition to using OR logits, we fine-tune the M2OR pre-trained GNN encoder.
        # we want to copy the weights from `GNN` of our M2OR model to `gnn` of model
        print(OR_model.gnn)

        for original_layer, new_layer in zip(OR_model.gnn.gnn_layers, model.gnn.gnn_layers):
            # Copying weights and biases for graph_conv
            new_layer.graph_conv.weight.data = original_layer.graph_conv.weight.data.clone()
            new_layer.graph_conv.bias.data = original_layer.graph_conv.bias.data.clone()

            # Copying weights and biases for res_connection (Linear layer)
            new_layer.res_connection.weight.data = original_layer.res_connection.weight.data.clone()
            new_layer.res_connection.bias.data = original_layer.res_connection.bias.data.clone()

            # If there are other parameters or buffers, they should also be copied similarly
        # Optionally, verify that the weights are the same
        print("Weights copied:")

        """
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
        """
    else:
        model = load_model(exp_config).to(args['device'])
        loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                         weight_decay=exp_config['weight_decay'])
        stopper = EarlyStopping(patience=exp_config['patience'],
                                filename=args['result_path'] + '/model.pth',
                                metric=args['metric'])
    

    exp_config['gnn_attended_feats'] = args['gnn_attended_feats']
    ## here we update n_tasks to be the number of tasks in the previous dataset (M2OR usually)
    exp_config.update({'n_tasks': args['prev_data_n_tasks']})
    ## Change model to GCN type to load M2OR model temporarily (super hacky)
    exp_config.update({'model': 'MolOR'})
    
    exp_config.update({'add_feat_size' : args['prot_dim']}) #args['prot_dim'] NOTE: NEED LESS HACKY WAY
    ## Load OR predictive model to generate OR preds as features
    exp_config['num_gnn_layers'] = 2 ## MolOR is 2 GNN layers
    exp_config['predictor_hidden_feats'] = 128 ## MolOR is 256 hidden feats
    exp_config['gnn_hidden_feats'] = 256 ## MolOR is 256 hidden feats
    OR_model = load_model(exp_config).to(args['device'])
    ## Now we load the model weights for the previously trained model, in this case Uniprot-M2OR GCN
    OR_checkpoint = torch.load(args['prev_model_path'] + '/model.pth', map_location=args['device'])
    OR_model.load_state_dict(OR_checkpoint['model_state_dict'])
    OR_model.eval()

    ## Get full GS_LF OR logits (5862, 845)
    full_OR_logits = None

    # generate OR logits if none on disk
    if full_OR_logits is None:
        args['model'] = "MolOR"
        print('Generating OR logits')
        #$ Now that the model weights are loaded, lets revert n_tasks back to the current dataset's number of tasks (GS-LF)
        full_OR_logits = torch.zeros(len(dataset), len(top_seqs)).cuda()

        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                idxs, smiles, bg, labels, masks, ids, node_masks = batch_data
                #print('seq embed shape')
                #print(seq_embeddings.shape)
                seq_masks = seq_masks.cuda()
                node_masks = node_masks.cuda()

                if len(smiles) == 1:
                    # Avoid potential issues with batch normalization
                    continue

                if batch_id % 4 == 0:
                    print('batch_id: ' + str(batch_id))

                labels, masks = labels.to(args['device']), masks.to(args['device'])
                ## Is it faster to iterate through each mol and generate logits for all 100 sequences by copying
                ## over graph features vs what we do now? Unlikely since torch.repeat is faster.
                #OR_logits = torch.zeros((len(smiles), seq_embeddings.shape[0])).cuda()
                for i in range(seq_embeddings.shape[0]):
                    ## Get i-th sequence embedding
                    seq_embed = seq_embeddings[i]
                    #print('seq embed shape out of dataloader')
                    # print what device seq_embed is on
                    seq_mask = seq_masks[i]
                    # Copy len(smiles) times into tensor of shape (len(smiles), seq_embed.shape)
                    seq_embed = seq_embed.repeat(len(smiles), 1, 1)
                    seq_mask = seq_mask.repeat(len(smiles), 1)
                    #print(predict_OR_feat(args, aux_model, bg, seq_embed, seq_mask, node_masks).shape)
                    #print('dimension of logits tensor index')
                    #print(train_OR_logits[batch_id, :, i].shape)
                    full_OR_logits[idxs, i] = predict_OR_feat(args, OR_model, bg, seq_embed, seq_mask, node_masks).squeeze(dim=1)
                #print(train_OR_logits)

            if args['prev_model_loss'] == "weighted_loss": # using m2or_cross_attention_batch_size_128_layernorm_weighted_loss
                torch.save(full_OR_logits, "data/datasets/weighted_loss_olfactory_subgenome_OR_logits.pt")
            else:
                torch.save(full_OR_logits, "data/datasets/olfactory_subgenome_OR_logits.pt")
            
            print('done GS_LF OR logit predictions')

    print(full_OR_logits.shape)

if __name__ == '__main__':
    import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #CUDA_LAUNCH_BLOCKING = "1"
    from argparse import ArgumentParser

    from utils import init_featurizer, mkdir_p, split_dataset, get_configure

    parser = ArgumentParser('Multi-label Binary Classification')
    """
    parser.add_argument('-d', '--dataset', choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21'],
                        help='Dataset to use')
    """
    
    parser.add_argument('-d', '--dataset', choices=['M2OR', 'GS_LF', 'GS_LF_OR'], default='M2OR',
                        help='Dataset to use (only M2OR and GS_LF are supported)')
    
    
    parser.add_argument('-mo', '--model', choices=['GCN', 'GAT', 'GCN_OR', 'Weave', 'MPNN', 'AttentiveFP',
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
    parser.add_argument('-c', '--curriculum', action='store_true')
    parser.add_argument('-esm', '--esm_version', choices=['650m', '3B'])
    parser.add_argument('-esm_rand', '--esm_random_weights', action='store_true', default = False)
    parser.add_argument('-mol2prot', '--mol2prot_dim', action='store_true', default = False, 
                        help= 'Before doing cross-attention, either map node_dim (usually 256) to prot_dim (usually 1280), \
                        or vice versa')
    parser.add_argument('-n_ORs', '--num_OR_logits', type=int, default=10)
    parser.add_argument('-prot', '--prot_dim', type=int, default=1280)
    parser.add_argument('-prev', '--prev_data_n_tasks', type=int, default=574,
                        help= "For passing OR logits as features, specify n_tasks of previous dataset to correctly load saved model.")
    parser.add_argument('-pmp', '--prev_model_path', type=str, default='M2OR_Uniprot_original_GCN',
                        help = 'For model to generate OR logits, specify path to trained model to correctly load model.')
    parser.add_argument('-pl', '--prev_model_loss', choices=['weighted_loss', 'unweighted_loss'], default='unweighted_loss',)
    ## Seeded as random_state = 42
    parser.add_argument('-s', '--split', choices=['scaffold', 'random'], default='scaffold',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-gnn_attend', '--gnn_attended_feats', type=int, default=None)
    parser.add_argument('-rp', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')
    seeds = [32]

    for seed in seeds:
        args['seed'] = seed
        print('SEED NO: ' + str(seed))
        torch.manual_seed(seed)

        args = init_featurizer(args)
        mkdir_p(args['result_path'])
        smiles_to_g = SMILESToBigraph(add_self_loop=True, node_featurizer=args['node_featurizer'],
                                    edge_featurizer=args['edge_featurizer'])
        
        if args['dataset'] == 'M2OR':
            from data.m2or import M2OR
            dataset = M2OR(smiles_to_graph=smiles_to_g, 
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'],
                        preprocess=args['preprocess'])
        elif args['dataset'] == 'GS_LF':
            from data.m2or import GS_LF
            dataset = GS_LF(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
        elif args['dataset'] == 'GS_LF_OR':
            from data.m2or import GS_LF_OR
            dataset = GS_LF_OR(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'], load=True)
            #args['max_seq_len'] = dataset.max_seq_len
            ## arbitrarily pad to size 100 - REMOVE THIS
            #args['max_node_len'] = 100
            ## arbitrarily pad to size 100 to account for other datasets to do eval on
            args['max_node_len'] = dataset.max_node_len

        args['n_tasks'] = dataset.n_tasks
        train_set, val_set, test_set = split_dataset(args, dataset)
        exp_config = get_configure(args['model'], args['featurizer_type'], args['dataset'])
        
        # to generate OR logits, use this function
        #generate_OR_logits(args, exp_config, train_set, val_set, test_set, dataset)
        main(args, exp_config, dataset, train_set, val_set, test_set)
