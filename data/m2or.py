# SPDX-License-Identifier: Apache-2.0
#
# The Toxicology in the 21st Century initiative.

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url

from dgllife.data.csv_dataset import MoleculeCSVDataset
import torch
from utils import ROOT_DIR
import os
#.csv_dataset import MoleculeCSVDataset

__all__ = ['Tox21']
esm_model = None
esm_alphabet = None


class M2OR(MoleculeCSVDataset):
    """M2OR dataset.

    Each target result is a binary label.

    A common issue for multi-task prediction is that some datapoints are not labeled for
    all tasks. This is also the case for M2OR. In data pre-processing, we set non-existing
    labels to be 0 so that they can be placed in tensors and used for masking in loss computation.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.

    References:

        * [1] Matching olfactory receptor respones to odorants.

    Parameters
    ----------
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES string into a DGLGraph. If None, it uses
        :func:`dgllife.utils.SMILESToBigraph` by default.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to False.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    cache_file_path : str
        Path to the cached DGLGraphs, default to 'tox21_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> from dgllife.data import Tox21
    >>> from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer

    >>> smiles_to_g = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer())
    >>> dataset = Tox21(smiles_to_g)
    >>> # Get size of the dataset
    >>> len(dataset)
    7831
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph, labels, and masks
    >>> dataset[0]
    ('CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
     DGLGraph(num_nodes=16, num_edges=34,
              ndata_schemes={}
              edata_schemes={}),
     tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
     tensor([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.]))

    The dataset instance also contains information about molecule ids.

    >>> dataset.id[i]

    We can also get the id along with SMILES, DGLGraph, labels, and masks at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
     DGLGraph(num_nodes=16, num_edges=34,
              ndata_schemes={}
              edata_schemes={}),
     tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
     tensor([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.]),
     'TOX3021')

    To address the imbalance between positive and negative samples, we can re-weight
    positive samples for each task based on the training datapoints.

    >>> train_ids = torch.arange(1000)
    >>> dataset.task_pos_weights(train_ids)
    tensor([26.9706, 35.3750,  5.9756, 21.6364,  6.4404, 21.4500, 26.0000,  5.0826,
            21.4390, 14.7692,  6.1442, 12.4308])
    """
    def __init__(self, smiles_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 preprocess = 'original',
                 load=False,
                 log_every=1000,
                 cache_file_path='./m2or_dglgraph.bin',
                 n_jobs=1):
        #self._url = 'dataset/tox21.csv.gz'
        #data_path = get_download_dir() + '/tox21.csv.gz'
        if preprocess == 'original': ## original, raw dataset
            data_path = 'data/datasets/M2OR_OR_odorant_pairwise_no_mixtures_dgl.csv'
        elif preprocess == 'two_class': # Ensure that each tasks has datapts from both 0/1.
            data_path = 'data/datasets/M2OR_OR_odorant_pairwise_no_mixtures_dgl_two_class.csv'
        elif preprocess == 'filtered': ## Filtered st each task/receptor has at least 30 datapts.
            data_path = 'data/datasets/M2OR_OR_odorant_pairwise_no_mixtures_dgl_filtered.csv'
        elif preprocess == 'uniprot': ## labels are organized by Uniprot IDs, not unique sequences.
            data_path = 'data/datasets/M2OR_OR_odorant_pairwise_Uniprot_no_mixtures_dgl.csv'
        elif preprocess == 'mol_OR_pairs': ## using all mol-OR pairs, original training data for ICLR paper
            #data_path = '../dti_m2or/Pairs_M2OR_Original/full_data.csv'
            data_path = 'data/datasets/pairwise_original_m2or.csv'
        else:
            raise ValueError('Expect preprocess to be original, filtered, or two_class, got {}'.format(preprocess))
        #download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        df = pd.read_csv(data_path)

        if preprocess != 'mol_OR_pairs':
            self.id = df['InChi Key']
            df = df.drop(columns=['InChi Key'])
        else:
            self.id = df['smiles']

        self.load_full = False

        super(M2OR, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the ith datapoint for all tasks. T for the number of tasks.
        Tensor of dtype float32 and shape (T)
            Binary masks of the ith datapoint indicating the existence of labels for all tasks.
        str, optional
            Id for the ith datapoint, returned only when ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        

class M2OR_Pairs(MoleculeCSVDataset):
    """M2OR dataset.

    Each target result is a binary label.

    A common issue for multi-task prediction is that some datapoints are not labeled for
    all tasks. This is also the case for M2OR. In data pre-processing, we set non-existing
    labels to be 0 so that they can be placed in tensors and used for masking in loss computation.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.

    References:

        * [1] Matching olfactory receptor respones to odorants.

    Parameters
    ----------
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES string into a DGLGraph. If None, it uses
        :func:`dgllife.utils.SMILESToBigraph` by default.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to False.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    cache_file_path : str
        Path to the cached DGLGraphs, default to 'tox21_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> from dgllife.data import Tox21
    >>> from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer

    >>> smiles_to_g = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer())
    >>> dataset = Tox21(smiles_to_g)
    >>> # Get size of the dataset
    >>> len(dataset)
    7831
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph, labels, and masks
    >>> dataset[0]
    ('CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
     DGLGraph(num_nodes=16, num_edges=34,
              ndata_schemes={}
              edata_schemes={}),
     tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
     tensor([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.]))

    The dataset instance also contains information about molecule ids.

    >>> dataset.id[i]

    We can also get the id along with SMILES, DGLGraph, labels, and masks at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
     DGLGraph(num_nodes=16, num_edges=34,
              ndata_schemes={}
              edata_schemes={}),
     tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
     tensor([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.]),
     'TOX3021')

    To address the imbalance between positive and negative samples, we can re-weight
    positive samples for each task based on the training datapoints.

    >>> train_ids = torch.arange(1000)
    >>> dataset.task_pos_weights(train_ids)
    tensor([26.9706, 35.3750,  5.9756, 21.6364,  6.4404, 21.4500, 26.0000,  5.0826,
            21.4390, 14.7692,  6.1442, 12.4308])
    """
    def __init__(self, smiles_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 weighted_samples = False,
                 cross_attention = False,
                 esm_model = '650m',
                 esm_random_weights = False,
                 load_full = False,
                 log_every=1000,
                 cache_file_path='./m2or_dglgraph.bin',
                 max_node_len = 22,
                 n_jobs=1):
        if weighted_samples:
            data_path = 'data/datasets/M2OR_sample_weights_pairs.csv'
        else:
            data_path = 'data/datasets/M2OR_original_mol_OR_pairs.csv'
            ## JOIN data_path with ROOT_DIR
            data_path = ROOT_DIR + '/' + data_path
        df = pd.read_csv(data_path, sep=';')

        self.id =  df['mol_id'].astype(str) + '-' + df['seq_id'].astype(str)
        self.seq_id = df['seq_id'].astype(str)
        #self.data_quality = df['_DataQuality']
        
        # For now, generate embeddings ahead of time instead of on the fly since we don't have many unique sequences (1237 mutants)
        self.sequences_dict = df.groupby('seq_id').apply(lambda x: x['mutated_Sequence'].unique()).apply(pd.Series).to_dict()[0]
        sequences = list(self.sequences_dict.values())
        self.cross_attention = cross_attention
        seq_lst =  df['mutated_Sequence'].tolist()
        self.max_seq_len = len(max(sequences, key=len))
        self.seq_mask = torch.zeros((len(df), self.max_seq_len))
        for i in range(len(df)):
            self.seq_mask[i, :len(seq_lst[i])] = 1
        if cross_attention:
            ## Define 2d torch tensor of shape ()
            for i in range(len(sequences)): ## pad sequence to max sequence length
                sequences[i] += "<pad>"*(self.max_seq_len - len(sequences[i]))
            #print(sequences)
            if os.path.exists('data/datasets/{}_per_residue_seq_embeddings.npy'.format(esm_model)):
                seq_embeddings = torch.tensor(np.load('data/datasets/{}_seq_embeddings.npy'.format(esm_model)))
            else:
                seq_embeddings = esm_embed(sequences, per_residue=True, random_weights=esm_random_weights, esm_model_version = esm_model) ## output shape: (batch_size, max_seq_len, embedding_dim)
                np.save('data/datasets/{}_per_residue_seq_embeddings.npy'.format(esm_model), seq_embeddings)
        else:
            if os.path.exists('data/datasets/{}_seq_embeddings.npy'.format(esm_model)):
                seq_embeddings = torch.tensor(np.load('data/datasets/{}_seq_embeddings.npy'.format(esm_model)))
            else:
                seq_embeddings = esm_embed(sequences, random_weights=esm_random_weights, esm_model_version = esm_model) ## output shape: (batch_size, embedding_dim)
                np.save('data/datasets/{}_seq_embeddings.npy'.format(esm_model), seq_embeddings)
        ## define dictionary where keys are from sequences_dict, and values are from self.seq_embeddings
        self.seq_embeddings_dict = dict(zip(self.sequences_dict.keys(), seq_embeddings))
        if weighted_samples:
            self.sample_weights = torch.tensor(df['sample_weight'].astype(float))
            df = df.drop(columns={'weight_pair_imbalance', 'weight_class', 'weight_quality', 'sample_weight'})
        else:
            import numpy as np
            self.sample_weights = torch.tensor(pd.Series(1.0, index=np.arange(len(df)), name='orders'))
        
        df['smiles'] = df['canonicalSMILES']
        df = df.drop(columns={'mol_id', 'seq_id', '_DataQuality', 'num_unique_value_screen', 'mutated_Sequence'})
        df = df[['smiles', 'Responsive']]

        self.load_full = load_full
        #print(self.seq_id)

        super(M2OR_Pairs, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]
        
        ## create 
        if max_node_len == 22:
            self.max_node_len = max([g.number_of_nodes() for g in self.graphs])
        else:
            self.max_node_len = max_node_len
        self.graph_mask = torch.zeros((len(self.graphs), self.max_node_len))
        for idx in range(len(self.graphs)):
            self.graph_mask[idx, :self.graphs[idx].num_nodes()] = 1

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the ith datapoint for all tasks. T for the number of tasks.
        Tensor of dtype float32 and shape (T)
            Binary masks of the ith datapoint indicating the existence of labels for all tasks.
        str, optional
            Id for the ith datapoint, returned only when ``self.load_full`` is True.
        """
        if self.load_full:
            if self.cross_attention:
                return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item], self.seq_id[item], \
                   self.sequences_dict[self.seq_id[item]], \
                   self.seq_embeddings_dict[self.seq_id[item]], \
                   self.sample_weights[item], self.seq_mask[item], self.graph_mask[item]
            else:
                return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item], self.seq_id[item], self.sequences_dict[self.seq_id[item]], self.seq_embeddings_dict[self.seq_id[item]], self.sample_weights[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


def setup_esm(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), random_weights=False, esm_model_version = '650m'):
    import esm
    # Load ESM-2 model
    global esm_model
    global esm_alphabet
    if esm_model is None:
        print('loading esm model...')
        if esm_model_version == '650m':
            print('loading ESM 650M model')
            esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D() ######################################
        else:
            print('loading ESM 3B model')
            esm_model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D() ######################################
        esm_model.eval()  # disables dropout for deterministic results
        esm_model.to(device)
        print('done loading esm model')
    
    if random_weights:
        reinitialize_weights(esm_model)
    return esm_model, esm_alphabet

def reinitialize_weights(model):
    import torch.nn.init as init
    seed = 42
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if 'embed' in name:  # Re-initialize embeddings
            if len(param.size()) > 1:
                init.normal_(param.data, mean=0.0, std=0.02)
        elif 'weight' in name:  # Re-initialize weights of linear layers
            if len(param.size()) > 1:
                init.xavier_normal_(param.data)
        elif 'bias' in name:  # Re-initialize biases of linear layers
            init.constant_(param.data, 0.0)


def esm_embed(sequences, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), per_residue = False, random_weights = False, esm_model_version = '650m'):
    # get the embeddings for a list of sequences. Code is copied from ESM github readme
    assert isinstance(sequences, list)
    esm_model, esm_alphabet = setup_esm(random_weights=random_weights, esm_model_version = esm_model_version)
    batch_converter = esm_alphabet.get_batch_converter()
    
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    sequence_representations = []
    sequence_chunks = list(divide_chunks(sequences, 5))
    for sequence_chunk in sequence_chunks:
        data = []
        for i, sequence in enumerate(sequence_chunk):
            assert ' ' not in sequence
            ##Hack: make sure not counting length > 1600 because of a bunch of padding
            if len(sequence) > 1600 and '<pad>' not in sequence:
                print('trimming sequence to 1600 amino acids max')
                sequence = sequence[0:1600]
            data.append(('protein' + str(i), sequence))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.inference_mode():
            results = esm_model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        #print(token_representations.shape)

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        #print (batch_lens)
        if per_residue:
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : -1].detach().cpu().numpy())
        else:
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).detach().cpu().numpy())
    print('done embedding sequences')
    return sequence_representations

def esm_embed_2(sequences, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), per_residue = False, random_weights = False, esm_model_version = '650m'):
    # get the embeddings for a list of sequences. Code is copied from ESM github readme
    assert isinstance(sequences, list)
    esm_model, esm_alphabet = setup_esm(random_weights=random_weights, esm_model_version = esm_model_version)
    batch_converter = esm_alphabet.get_batch_converter()
    max_seq_len = 705
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    if esm_model_version == '650m':
        dim = 1280
    else:
        dim = 2560
    if per_residue:
        sequence_representations = torch.zeros(len(sequences), max_seq_len, dim)
    else:
        sequence_representations = torch.zeros(len(sequences), dim)
    sequence_chunks = list(divide_chunks(sequences, 5))
    count = 0
    for sequence_chunk in sequence_chunks:
        data = []
        for i, sequence in enumerate(sequence_chunk):
            assert ' ' not in sequence
            ##Hack: make sure not counting length > 1600 because of a bunch of padding
            if len(sequence) > 1600 and '<pad>' not in sequence:
                print('trimming sequence to 1600 amino acids max')
                sequence = sequence[0:1600]
            data.append(('protein' + str(i), sequence))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.inference_mode():
            results = esm_model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        #print(token_representations.shape)

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        #print (batch_lens)
        if per_residue:
            for i, tokens_len in enumerate(batch_lens):
                print(len(sequence_chunk[i]))
                print(token_representations[i, 1 : -1].shape)
                print(sequence_representations[count].shape)
                sequence_representations[count] = token_representations[i, 1 : -1].detach()#.cpu()#.numpy()
                #sequence_representations.append(token_representations[i, 1 : -1].detach().cpu().numpy())
                count +=1
        else:
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations[count] = token_representations[i, 1 : tokens_len - 1].mean(0).detach()
                count +=1
                #sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).detach().cpu().numpy())
    print('done embedding sequences')
    return sequence_representations



def get_weight_cols(df):
    ## Presumes df haS columns mol_id, seq_id, Responsiveness, and _dataQuality. 
    ## For data quality weights:
    ## Using Mainland et. al (2015) data for primary, secondary, and tertiary screening, with the conditional probabilities for each
    ## positive or negative primary/secondary screen being truly positive or negative in the tertiary screen to reweigh loss
    
    pos_weight = df['Responsive'].value_counts()[0] / df['Responsive'].value_counts()[1]
    import numpy as np
    k = 50

    for i in range(len(df)):
        if df.loc [i, '_DataQuality'] == 'ec50':
            df.loc[i, 'weight_quality'] = 1
        elif df.loc[i, '_DataQuality'] == 'primaryScreening':
            if df.loc[i, 'Responsive'] == 1:
                df.loc[i, 'weight_quality'] = 0.4
            else:
                df.loc[i, 'weight_quality'] = 0.69
        elif df.loc[i, '_DataQuality'] == 'secondaryScreening':
            if df.loc[i, 'Responsive'] == 1:
                df.loc[i, 'weight_quality'] = 0.72
            else:
                df.loc[i, 'weight_quality'] = 0.77

        if df.loc [i, 'Responsive'] == 1:
            df.loc[i, 'weight_class'] = pos_weight
        else:
            df.loc[i, 'weight_class'] = 1

        curr_receptor = df.iloc[i]['seq_id']
        curr_mol = df.iloc[i]['mol_id']
        num_mols = df[df['seq_id'] == curr_receptor]['mol_id'].shape[0]
        num_receptors = df[df['mol_id'] == curr_mol]['seq_id'].shape[0]
        
        df.loc[i, 'weight_pair_imbalance'] = np.log(1 + k/2 * (1/num_mols + 1/num_receptors))

    df['weight_pair_imbalance'] = df['weight_pair_imbalance'].astype(float)
    
    df.loc[i, 'sample_weight'] = df.iloc[i]['weight_quality'] * df.iloc[i]['weight_class'] * df.iloc[i]['weight_pair_imbalance']
    
    return df

        
    
class GS_LF(MoleculeCSVDataset):
    """Goodscents-Leffingwell Joint dataset.

    Each target result is a binary label.

    A common issue for multi-task prediction is that some datapoints are not labeled for
    all tasks. This is also the case for GS-LF. In data pre-processing, we set non-existing
    labels to be 0 so that they can be placed in tensors and used for masking in loss computation.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.

    References:

        * [1] Matching olfactory receptor respones to odorants.

    Parameters
    ----------
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES string into a DGLGraph. If None, it uses
        :func:`dgllife.utils.SMILESToBigraph` by default.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to False.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    cache_file_path : str
        Path to the cached DGLGraphs, default to 'tox21_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> from dgllife.data import Tox21
    >>> from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer

    >>> smiles_to_g = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer())
    >>> dataset = Tox21(smiles_to_g)
    >>> # Get size of the dataset
    >>> len(dataset)
    7831
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph, labels, and masks
    >>> dataset[0]
    ('CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
     DGLGraph(num_nodes=16, num_edges=34,
              ndata_schemes={}
              edata_schemes={}),
     tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
     tensor([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.]))

    The dataset instance also contains information about molecule ids.

    >>> dataset.id[i]

    We can also get the id along with SMILES, DGLGraph, labels, and masks at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
     DGLGraph(num_nodes=16, num_edges=34,
              ndata_schemes={}
              edata_schemes={}),
     tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
     tensor([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.]),
     'TOX3021')

    To address the imbalance between positive and negative samples, we can re-weight
    positive samples for each task based on the training datapoints.

    >>> train_ids = torch.arange(1000)
    >>> dataset.task_pos_weights(train_ids)
    tensor([26.9706, 35.3750,  5.9756, 21.6364,  6.4404, 21.4500, 26.0000,  5.0826,
            21.4390, 14.7692,  6.1442, 12.4308])
    """
    def __init__(self, smiles_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 smiles_type = 'canonical',
                 load=False,
                 log_every=1000,
                 cache_file_path='./gs_lf_dglgraph.bin',
                 n_jobs=1):
        from rdkit import Chem
        
        ##TODO: - add option to process isomeric SMILES and benchmark both
        
        #self._url = 'dataset/tox21.csv.gz'
        #data_path = get_download_dir() + '/tox21.csv.gz'
        #download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        data_path = 'data/datasets/NaNs_GS_LF_isomeric_SMILES_dedup_odor_filtered.csv'
        df = pd.read_csv(data_path)
        self.id = df['CID']

        df = df.drop(columns=['Stimulus', 'CID', 'IUPACName', 'MolecularWeight', 'name'])  
        if smiles_type == 'canonical':
            iso_smiles = df['IsomericSMILES'].tolist()
            df['smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in iso_smiles]
            # Move column 'smiles' to front
            df = df[['smiles'] + [col for col in df.columns if col != 'smiles']]
            df = df.drop(columns=['IsomericSMILES'])
        self.load_full = False

        super(GS_LF, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the ith datapoint for all tasks. T for the number of tasks.
        Tensor of dtype float32 and shape (T)
            Binary masks of the ith datapoint indicating the existence of labels for all tasks.
        str, optional
            Id for the ith datapoint, returned only when ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        
        
class GS_LF_OR(MoleculeCSVDataset):
    """Goodscents-Leffingwell Joint dataset, with OR embeddings from top 100 observed sequences in M2OR dataset.

    Each target result is a binary label.

    A common issue for multi-task prediction is that some datapoints are not labeled for
    all tasks. This is also the case for GS-LF. In data pre-processing, we set non-existing
    labels to be 0 so that they can be placed in tensors and used for masking in loss computation.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.

    References:

        * [1] Matching olfactory receptor respones to odorants.

    Parameters
    ----------
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES string into a DGLGraph. If None, it uses
        :func:`dgllife.utils.SMILESToBigraph` by default.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to False.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    cache_file_path : str
        Path to the cached DGLGraphs, default to 'tox21_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> from dgllife.data import Tox21
    >>> from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer

    >>> smiles_to_g = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer())
    >>> dataset = Tox21(smiles_to_g)
    >>> # Get size of the dataset
    >>> len(dataset)
    7831
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph, labels, and masks
    >>> dataset[0]
    ('CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
     DGLGraph(num_nodes=16, num_edges=34,
              ndata_schemes={}
              edata_schemes={}),
     tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
     tensor([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.]))

    The dataset instance also contains information about molecule ids.

    >>> dataset.id[i]

    We can also get the id along with SMILES, DGLGraph, labels, and masks at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
     DGLGraph(num_nodes=16, num_edges=34,
              ndata_schemes={}
              edata_schemes={}),
     tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
     tensor([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.]),
     'TOX3021')

    To address the imbalance between positive and negative samples, we can re-weight
    positive samples for each task based on the training datapoints.

    >>> train_ids = torch.arange(1000)
    >>> dataset.task_pos_weights(train_ids)
    tensor([26.9706, 35.3750,  5.9756, 21.6364,  6.4404, 21.4500, 26.0000,  5.0826,
            21.4390, 14.7692,  6.1442, 12.4308])
    """
    def __init__(self, smiles_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 smiles_type = 'canonical',
                 load=False,
                 log_every=1000,
                 cache_file_path='./gs_lf_dglgraph.bin',
                 n_jobs=1):
        from rdkit import Chem
        
        ##TODO: - add option to process isomeric SMILES and benchmark both
        
        #self._url = 'dataset/tox21.csv.gz'
        #data_path = get_download_dir() + '/tox21.csv.gz'
        #download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        #mol_OR = pd.read_csv('data/datasets/M2OR_original_mol_OR_pairs.csv', sep = ';')
        #top_100_seqs = mol_OR['mutated_Sequence'].value_counts()[0:100].keys().tolist()
        #self.max_seq_len = len(max(top_100_seqs, key=len))        
        #self.seq_mask = torch.zeros((len(top_100_seqs), self.max_seq_len))
        #for i in range(len(top_100_seqs)):
        #    self.seq_mask[i, :len(top_100_seqs[i])] = 1
        #    top_100_seqs[i] += "<pad>"*(self.max_seq_len - len(top_100_seqs[i]))
        
        #self.seq_embeddings = esm_embed(top_100_seqs, per_residue=True, random_weights=False, esm_model_version = '650m') ## output shape: (100, max_seq_len, embedding_dim)
        #print(len(self.seq_embeddings))#.shape)
        data_path = 'data/datasets/NaNs_GS_LF_isomeric_SMILES_dedup_odor_filtered.csv'
        df = pd.read_csv(data_path)
        self.id = df['CID']

        df = df.drop(columns=['Stimulus', 'CID', 'IUPACName', 'MolecularWeight', 'name'])
        if smiles_type == 'canonical':
            iso_smiles = df['IsomericSMILES'].tolist()
            df['smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in iso_smiles]
            # Move column 'smiles' to front
            df = df[['smiles'] + [col for col in df.columns if col != 'smiles']]
            df = df.drop(columns=['IsomericSMILES'])
        self.load_full = load

        super(GS_LF_OR, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]
        
        ## create 
        self.max_node_len = max([g.number_of_nodes() for g in self.graphs])
        self.graph_mask = torch.zeros((len(self.graphs), self.max_node_len))
        for idx in range(len(self.graphs)):
            self.graph_mask[idx, :self.graphs[idx].num_nodes()] = 1


    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the ith datapoint for all tasks. T for the number of tasks.
        Tensor of dtype float32 and shape (T)
            Binary masks of the ith datapoint indicating the existence of labels for all tasks.
        str, optional
            Id for the ith datapoint, returned only when ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item], self.graph_mask[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        
        