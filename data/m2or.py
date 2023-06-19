# SPDX-License-Identifier: Apache-2.0
#
# The Toxicology in the 21st Century initiative.

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url

from dgllife.data.csv_dataset import MoleculeCSVDataset
#.csv_dataset import MoleculeCSVDataset

__all__ = ['Tox21']

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
        else:
            raise ValueError('Expect preprocess to be original, filtered, or two_class, got {}'.format(preprocess))
        #download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        df = pd.read_csv(data_path)
        self.id = df['InChi Key']

        df = df.drop(columns=['InChi Key'])
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
        
        
        
        