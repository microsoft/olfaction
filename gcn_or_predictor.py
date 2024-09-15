import torch
import torch.nn as nn

from dgllife.model.model_zoo.mlp_predictor import MLPPredictor
from dgllife.model.gnn.gcn import GCN
from dgllife.model.gnn.gatv2 import GATv2
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from torch.nn.functional import scaled_dot_product_attention
import dgl
import numpy as np

class GCNJointPredictor(nn.Module):
    """GCN-based model for regression and classification on graphs with two heads;
    
    a) Head 1 will take as input mol embeddings and out OR logits, which are then fed as input to Head 2,
    b) Head 2 will take as input mol embeddings and head 1's outputted OR logits, and output percept logits.
    """
    
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 add_feats = False,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=[574, 152], predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNJointPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.add_feats = add_feats
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        self.predict_ORs = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                        n_tasks[0], predictor_dropout)
        
        if add_feats:
            self.predict_scent = MLPPredictor(2 * gnn_out_feats + n_tasks[0], predictor_hidden_feats,
                n_tasks[1], predictor_dropout)
        else:
            self.predict_scent = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks[1], predictor_dropout)
    def forward(self, bg, feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization
        add_feats : FloatTensor of shape (B, feat_tasks)
            * B for the number of graphs in the batch
            * feat_tasks is the number of tasks for the additional features
              (predicted logits for OR activations in this case)

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        ## Concatenate OR features to graph_feats before prediction
        OR_logits = self.predict_ORs(graph_feats)
        if self.add_feats is True:
            graph_feats_scent = torch.cat((graph_feats, OR_logits), dim=1)
        else:
            graph_feats_scent = graph_feats    
        
        return OR_logits, self.predict_scent(graph_feats_scent)

    
# pylint: disable=W0221
class GCNORPredictor(nn.Module):
    """GCN-based model for regression and classification on graphs with OR features appended
    before classification by MLP head.

    GCN is introduced in `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__. This model is based on GCN and can be used
    for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers. By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If None, no activation will be applied. If not None, ``activation[i]`` gives the
        activation function to be used for the i-th GCN layer. ``len(activation)`` equals
        the number of GCN layers. By default, ReLU is applied for all GCN layers.
    add_feats : int 
        Number of input OR features.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 add_feats = None,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNORPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        if add_feats:
            self.predict = MLPPredictor(2 * gnn_out_feats + add_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)
        else:
            self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats, add_feats = None):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization
        add_feats : FloatTensor of shape (B, feat_tasks)
            * B for the number of graphs in the batch
            * feat_tasks is the number of tasks for the additional features
              (predicted logits for OR activations in this case)

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        if add_feats.dim() > 2: #(n_samples, 1, 1280) --> (nsamples, 1280)
            add_feats = add_feats.squeeze(1)
        ## Concatenate OR features to graph_feats before prediction
        
        if add_feats is not None:
            graph_feats = torch.cat((graph_feats, add_feats), dim=1)
        
        return self.predict(graph_feats)


    

""" 
## TODO: 
- (done) Implement MLP layer that collapses ESM logits from size (R, D1) to (R, D2) where D2 = gnn_out_feats --> do so to construct query, key and value protein tensors.
- implement cross-attention beyween output of protein MLP layer and output of GCN (with self attention grease for molecular and protein emb first)
- plumbing to fit into GCNPredictor (new version of class)
- plumbing in classification_ESM.py to support new model

"""

class CrossAttention(nn.Module):
    
    """Cross-Attention Block of ligand-protein model, that takes in two 2d tensors for the molecular and protein embeddings, collapses them to the same
    dimension, and performs a cross-attention between them. Uses torch scaled dot product attention.
    
    Args:
        D1 (int): dimension of input protein tensor
        D2 (int): dimension of input mol tensor (usually smaller)
        mol2_prot (bool): If true, then linear Q, K, V tensors are of size D1 (mol --> prot dim).
    """

    def __init__(self, D1, D2, mol2prot = False):
        super(CrossAttention, self).__init__()

        # Define the trainable weight matrices for query, key, and value transformations
        if mol2prot: ## linear mapping to expand mol tensor to protein tensor size
            self.query_transform_tensor1 = nn.Linear(D1, D1)
            self.key_transform_tensor1 = nn.Linear(D1, D1)
            self.value_transform_tensor1 = nn.Linear(D1, D1)

            self.query_transform_tensor2 = nn.Linear(D2, D1)
            self.key_transform_tensor2 = nn.Linear(D2, D1)
            self.value_transform_tensor2 = nn.Linear(D2, D1)

            # Linear layer for aggregation
            self.linear1 = nn.Linear(D1, 1)
            self.linear2 = nn.Linear(D1, 1)
        else: ## original cross-attention experiment collapses protein tensor to mol tensor size
            self.query_transform_tensor1 = nn.Linear(D1, D2)
            self.key_transform_tensor1 = nn.Linear(D1, D2)
            self.value_transform_tensor1 = nn.Linear(D1, D2)

            self.query_transform_tensor2 = nn.Linear(D2, D2)
            self.key_transform_tensor2 = nn.Linear(D2, D2)
            self.value_transform_tensor2 = nn.Linear(D2, D2)

            # Linear layer for aggregation
            self.linear1 = nn.Linear(D2, 1)
            self.linear2 = nn.Linear(D2, 1)

    def scaled_attention_weights(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        #attention_weights = torch.softmax(scores, dim=-1) ## try relu here, then softmax at the end w/ the MLP prediction head
        attention_weights = torch.relu(scores)
        #NOTE : run with softmax fix
        return attention_weights

    def gen_attn_maps(self, tensor1, tensor2, seq_mask, node_mask):
        ## Using seq_mask to mask out padding tokens in the protein sequence
        tensor1 = tensor1 * seq_mask[:, :, np.newaxis]
        
        ## Tensor2 is already masked out, as its filled in from an empty tensor object
        
        # Compute query, key, and value representations for both tensors
        query_tensor1 = self.query_transform_tensor1(tensor1)
        key_tensor1 = self.key_transform_tensor1(tensor1)
        value_tensor1 = self.value_transform_tensor1(tensor1)

        query_tensor2 = self.query_transform_tensor2(tensor2)
        key_tensor2 = self.key_transform_tensor2(tensor2)
        value_tensor2 = self.value_transform_tensor2(tensor2)

        prot_attention_maps = self.scaled_attention_weights(query_tensor1, key_tensor2, value_tensor2) ## outputs (batch_size, R, A)
        mol_attention_maps = self.scaled_attention_weights(query_tensor2, key_tensor1, value_tensor1) ## outputs (batch_size, A, R)

        return prot_attention_maps, mol_attention_maps

    def forward(self, tensor1, tensor2, seq_mask, node_mask):
        
        ## Using seq_mask to mask out padding tokens in the protein sequence
        tensor1 = tensor1 * seq_mask[:, :, np.newaxis] 
        
        ## Tensor2 is already masked out, as its filled in from an empty tensor object
        
        # Compute query, key, and value representations for both tensors
        query_tensor1 = self.query_transform_tensor1(tensor1)
        key_tensor1 = self.key_transform_tensor1(tensor1)
        value_tensor1 = self.value_transform_tensor1(tensor1)

        query_tensor2 = self.query_transform_tensor2(tensor2)
        key_tensor2 = self.key_transform_tensor2(tensor2)
        value_tensor2 = self.value_transform_tensor2(tensor2)

        # Compute cross-attention between tensor1 and tensor2
        #attended_values_tensor1 = self.scaled_dot_product_attention(query_tensor1, key_tensor2, value_tensor2) ## outputs (batch_size, R, D2)
        #attended_values_tensor2 = self.scaled_dot_product_attention(query_tensor2, key_tensor1, value_tensor1) ## outputs (batch_size, A, D2)
        attended_values_tensor1 = scaled_dot_product_attention(query_tensor1, key_tensor2, value_tensor2)
        attended_values_tensor2 = scaled_dot_product_attention(query_tensor2, key_tensor1, value_tensor1)
        
        #print(attended_values_tensor1.shape)
        #print(attended_values_tensor2.shape)
        #attended_values_tensor2 = self.scaled_dot_product_attention(query_tensor2, key_tensor1, value_tensor1)

        # Apply Linear for aggregation
        fixed_size_tensor1 = self.linear1(attended_values_tensor1).squeeze(-1) ## outputs (batch_size, R)
        fixed_size_tensor2 = self.linear2(attended_values_tensor2).squeeze(-1) ## outputs (batch_size, A)
        #print('atom linear tensor')
        #print(fixed_size_tensor2)

        ## TODO: might make more sense to pad before activation.
        ## Set the padded residues and nodes to neg infinity before MLP predictor (uses softmax to set to 0)
        fixed_size_tensor1[seq_mask == 0] = 0 #-torch.inf Stop setting to neg inf due to NaN logits
        fixed_size_tensor2[node_mask == 0] = 0 # -torch.inf Stop setting to neg inf due to NaN logits

        # Apply softmax on the fixed size tensors such that the sum of residues (R) or atoms (A) is 1
        #softmax_fixed_size_tensor1 = torch.softmax(fixed_size_tensor1, dim=1) # sum of residues is 1
        #softmax_fixed_size_tensor2 = torch.softmax(fixed_size_tensor2, dim=1) # sum of atoms is 1
        
        tensor1 = tensor1.transpose(1, 2) ## transpose such that dimensions are (batch_size, D1, R)
        protein_vec = torch.einsum('ijk,ik->ij', tensor1, fixed_size_tensor1) ## outputs (batch_size, D1)
        
        tensor2 = tensor2.transpose(1, 2)
        mol_vec = torch.einsum('ijk, ik->ij', tensor2, fixed_size_tensor2) ## outputs (batch_size, D2)
        
        output_vec = torch.cat((protein_vec, mol_vec), dim=1)
        ## concat into output_vector (size: (batch_size, prot_seq_len + node_len))
        # output_vec = torch.cat((fixed_size_tensor1, fixed_size_tensor2), dim=1)
        return output_vec    

        # NOTE: code below does mean aggregation over residue + atoms, before concat
        # NOTE: below, we're temporarily trying to use the mean of the attended values as the fixed size tensor 
        """
        fixed_size_tensor1 = attended_values_tensor1.mean(dim=1) # B x D1
        fixed_size_tensor2 = attended_values_tensor2.mean(dim=1) # B x D1 (assuming projection to prot dim space)
        """
        output_vec = torch.cat((fixed_size_tensor1, fixed_size_tensor2), dim=1)
        ## concat into output_vector (size: (batch_size, prot_seq_len + node_len))
        # output_vec = torch.cat((fixed_size_tensor1, fixed_size_tensor2), dim=1) # now the output_vec is size (1, 2* D1)
        return output_vec
    
class OdorantReceptorCrossAttention(nn.Module):
    
    """Cross-Attention Block of ligand-protein model, that takes in two 2d tensors for the molecular and protein embeddings, collapses them to the same
    dimension, and performs a cross-attention between them. 
    
    Args:
        D1 (int): dimension of input protein tensor
        D2 (int): dimension of input mol tensor (usually smaller)
        mol2_prot (bool): If true, then linear Q, K, V tensors are of size D1 (mol --> prot dim).
    """

    def __init__(self, D1, D2, mol2prot = False):
        super(OdorantReceptorCrossAttention, self).__init__()

        # Define the trainable weight matrices for query, key, and value transformations
        if mol2prot: ## linear mapping to expand mol tensor to protein tensor size
            self.query_transform_tensor1 = nn.Linear(D1, D1)
            self.key_transform_tensor1 = nn.Linear(D1, D1)
            self.value_transform_tensor1 = nn.Linear(D1, D1)

            self.query_transform_tensor2 = nn.Linear(D2, D1)
            self.key_transform_tensor2 = nn.Linear(D2, D1)
            self.value_transform_tensor2 = nn.Linear(D2, D1)

            # Linear layer for aggregation
            self.linear1 = nn.Linear(D1, 1)
            self.linear2 = nn.Linear(D1, 1)
        else: ## original cross-attention experiment collapses protein tensor to mol tensor size
            self.query_transform_tensor1 = nn.Linear(D1, D2)
            self.key_transform_tensor1 = nn.Linear(D1, D2)
            self.value_transform_tensor1 = nn.Linear(D1, D2)

            self.query_transform_tensor2 = nn.Linear(D2, D2)
            self.key_transform_tensor2 = nn.Linear(D2, D2)
            self.value_transform_tensor2 = nn.Linear(D2, D2)

            # Linear layer for aggregation
            self.linear1 = nn.Linear(D2, 1)
            self.linear2 = nn.Linear(D2, 1)
        
    def scaled_dot_product_attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        #attention_weights = torch.softmax(scores, dim=-1) ## try relu here, then softmax at the end w/ the MLP prediction head
        attention_weights = torch.relu(scores) ## TODO: - visualize attention weights here
        #NOTE : run with softmax fix
        attended_values = torch.matmul(attention_weights, value)
        return attended_values

    def scaled_attention_weights(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        #attention_weights = torch.softmax(scores, dim=-1) ## try relu here, then softmax at the end w/ the MLP prediction head
        attention_weights = torch.relu(scores)
        #NOTE : run with softmax fix
        return attention_weights

    def gen_attn_maps(self, tensor1, tensor2, seq_mask, node_mask):
        ## Using seq_mask to mask out padding tokens in the protein sequence
        tensor1 = tensor1 * seq_mask[:, :, np.newaxis]
        
        ## Tensor2 is already masked out, as its filled in from an empty tensor object
        
        # Compute query, key, and value representations for both tensors
        query_tensor1 = self.query_transform_tensor1(tensor1)
        key_tensor1 = self.key_transform_tensor1(tensor1)
        value_tensor1 = self.value_transform_tensor1(tensor1)

        query_tensor2 = self.query_transform_tensor2(tensor2)
        key_tensor2 = self.key_transform_tensor2(tensor2)
        value_tensor2 = self.value_transform_tensor2(tensor2)

        prot_attention_maps = self.scaled_attention_weights(query_tensor1, key_tensor2, value_tensor2) ## outputs (batch_size, R, A)
        mol_attention_maps = self.scaled_attention_weights(query_tensor2, key_tensor1, value_tensor1) ## outputs (batch_size, A, R)

        return prot_attention_maps, mol_attention_maps

    def forward(self, tensor1, tensor2, seq_mask, node_mask):
        
        ## Using seq_mask to mask out padding tokens in the protein sequence
        tensor1 = tensor1 * seq_mask[:, :, np.newaxis] 
        
        ## Tensor2 is already masked out, as its filled in from an empty tensor object
        
        # Compute query, key, and value representations for both tensors
        query_tensor1 = self.query_transform_tensor1(tensor1)
        key_tensor1 = self.key_transform_tensor1(tensor1)
        value_tensor1 = self.value_transform_tensor1(tensor1)

        query_tensor2 = self.query_transform_tensor2(tensor2)
        key_tensor2 = self.key_transform_tensor2(tensor2)
        value_tensor2 = self.value_transform_tensor2(tensor2)

        # Compute cross-attention between tensor1 and tensor2
        attended_values_tensor1 = self.scaled_dot_product_attention(query_tensor1, key_tensor2, value_tensor2) ## outputs (batch_size, R, D2)
        attended_values_tensor2 = self.scaled_dot_product_attention(query_tensor2, key_tensor1, value_tensor1) ## outputs (batch_size, A, D2)
        #print(attended_values_tensor1.shape)
        #print(attended_values_tensor2.shape)
        #attended_values_tensor2 = self.scaled_dot_product_attention(query_tensor2, key_tensor1, value_tensor1)

        # Apply Linear for aggregation
        fixed_size_tensor1 = self.linear1(attended_values_tensor1).squeeze(-1) ## outputs (batch_size, R)
        fixed_size_tensor2 = self.linear2(attended_values_tensor2).squeeze(-1) ## outputs (batch_size, A)
        #print('atom linear tensor')
        #print(fixed_size_tensor2)

        ## TODO: might make more sense to pad before activation.
        ## Set the padded residues and nodes to neg infinity before MLP predictor (uses softmax to set to 0)
        fixed_size_tensor1[seq_mask == 0] = 0 #-torch.inf Stop setting to neg inf due to NaN logits
        fixed_size_tensor2[node_mask == 0] = 0 # -torch.inf Stop setting to neg inf due to NaN logits

        #NOTE: doesn't perform well. Idea: Apply softmax on the fixed size tensors such that the sum of residues (R) or atoms (A) is 1
        #softmax_fixed_size_tensor1 = torch.softmax(fixed_size_tensor1, dim=1) # sum of residues is 1
        #softmax_fixed_size_tensor2 = torch.softmax(fixed_size_tensor2, dim=1) # sum of atoms is 1
        
        tensor1 = tensor1.transpose(1, 2) ## transpose such that dimensions are (batch_size, D1, R)
        protein_vec = torch.einsum('ijk,ik->ij', tensor1, fixed_size_tensor1) ## outputs (batch_size, D1)
        
        tensor2 = tensor2.transpose(1, 2)
        mol_vec = torch.einsum('ijk, ik->ij', tensor2, fixed_size_tensor2) ## outputs (batch_size, D2)
        
        output_vec = torch.cat((protein_vec, mol_vec), dim=1)
        ## concat into output_vector (size: (batch_size, prot_seq_len + node_len))
        # output_vec = torch.cat((fixed_size_tensor1, fixed_size_tensor2), dim=1)
        return output_vec
    

    
class MolORPredictor(nn.Module):
    """GCN-based model for regression and classification on graphs with cross-attention layer
    before classification by MLP head.

    GCN is introduced in `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__. This model is based on GCN and can be used
    for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers. By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If None, no activation will be applied. If not None, ``activation[i]`` gives the
        activation function to be used for the i-th GCN layer. ``len(activation)`` equals
        the number of GCN layers. By default, ReLU is applied for all GCN layers.
    add_feats : int 
        Number of input OR features.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 add_feats = None, prot_feats = 1280, gnn_attended_feats = None, max_seq_len = 705, max_node_len = 22,
                 mol2_prot = False,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(MolORPredictor, self).__init__()
        torch.autograd.set_detect_anomaly(True)

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout
        self.max_node_len = max_node_len

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        #self.readout = WeightedSumAndMax(gnn_out_feats)
        
        #self.cross_attn = CrossAttention_2(prot_feats, gnn_out_feats, mol2prot = mol2_prot)
        # NOTE: trying with torch implementation
        self.cross_attn = OdorantReceptorCrossAttention(prot_feats, gnn_out_feats, mol2prot = mol2_prot)

        gnn_attended_feats = self.gnn.hidden_feats[-1] if gnn_attended_feats is None else gnn_attended_feats # output dimension of mol may differ
        
        self.predict = MLPPredictor(prot_feats + gnn_attended_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)
        
        # Layernorms before and after cross-attention
        self.prot_norm = nn.LayerNorm(prot_feats)
        self.mol_norm = nn.LayerNorm(gnn_out_feats)
        self.feat_norm = nn.LayerNorm(prot_feats + gnn_attended_feats)
        
        #self.predict = MLPPredictor(max_seq_len + max_node_len, predictor_hidden_feats, 
        #                            n_tasks, predictor_dropout)
        """
        if add_feats:
            self.predict = MLPPredictor(2 * gnn_out_feats + add_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)
        else:
            self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)
        """
    def forward(self, bg, feats, add_feats = None, seq_mask = None, node_mask = None, device = None):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization
        add_feats : FloatTensor of shape (B, n_residues, emb_dim)
            * B for the number of graph-OR pairs in the batch
            * n_residues is the number of residues in the protein sequence (padded to max_seq_len)
            * emb_dim is the embedding dimension of the protein sequence
        seq_mask : FloatTensor of shape (B, n_residues)
        node_mask : FloatTensor of shape (B, N)
        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        #print(bg)
        #print(feats)
        node_feats = self.gnn(bg, feats) ## problem causing NaNs is here
        #print('node feats')
        #print(node_feats)
        
        ## feed logits into bg for us to unbatch and index
        ## into correct graphs
        #bg.ndata['logits'] = node_feats
        
        ## Unbatch a batched graph
        graphs = dgl.unbatch(bg)
        batch_node_feats = torch.zeros((len(graphs), self.max_node_len, node_feats.shape[1]))
        
        ## fill in batch_node feats with node feats from each graph
        counter = 0
        for i in range(len(graphs)):
            n_nodes = graphs[i].num_nodes()
            batch_node_feats[i][:n_nodes] = node_feats[counter:n_nodes + counter]
            counter+=n_nodes
            #batch_node_feats[i][:graphs[i].num_nodes()] = graphs[i].ndata['logits']
        
        #graph_feats = self.readout(bg, node_feats)
        #print('post readout graph feats')
        #print(graph_feats)
        ## Concatenate OR features to graph_feats before prediction
        ## outputs a tensor of shape (B, max_seq_len)
        
        ## Now, batch_node_feats is a padded batch of 2d tensors of node-level features.
        ## We pad the sequence tensor in the cross_attn forward pass
        
        ## set add_feats, batch_node_feats to cuda
        if torch.cuda.is_available() and device is not None:
            add_feats = add_feats.to(device)
            batch_node_feats = batch_node_feats.to(device)
            #add_feats = add_feats.cuda()
            #batch_node_feats = batch_node_feats.cuda()
        
        #print("ADD FEATS:") 
        #print(add_feats)
        #print("BATCH NODE FEATS:")
        #print(batch_node_feats)
        #print('seq emb shape:')
        #print(add_feats.shape)

        # LayerNorm on minibatch of per-residue protein embeddings, per-node embeddings
        add_feats = self.prot_norm(add_feats)
        batch_node_feats = self.mol_norm(batch_node_feats)
        
        graph_feats = self.cross_attn(add_feats, batch_node_feats, seq_mask, node_mask)
        #if add_feats is not None:
        #    graph_feats = torch.cat((graph_feats, add_feats), dim=1)
        # LayerNorm on final weighted joint protein-molecule embeddings
        graph_feats = self.feat_norm(graph_feats)
        
        return self.predict(graph_feats)

    def generate_attention_maps(self, bg, feats, add_feats = None, seq_mask = None, node_mask = None, device = None):
        node_feats = self.gnn(bg, feats) ## problem causing NaNs is here
        #print('node feats')
        #print(node_feats)
        
        ## feed logits into bg for us to unbatch and index
        ## into correct graphs
        #bg.ndata['logits'] = node_feats
        
        ## Unbatch a batched graph
        graphs = dgl.unbatch(bg)
        batch_node_feats = torch.zeros((len(graphs), self.max_node_len, node_feats.shape[1]))
        
        ## fill in batch_node feats with node feats from each graph
        counter = 0
        for i in range(len(graphs)):
            n_nodes = graphs[i].num_nodes()
            batch_node_feats[i][:n_nodes] = node_feats[counter:n_nodes + counter]
            counter+=n_nodes
            #batch_node_feats[i][:graphs[i].num_nodes()] = graphs[i].ndata['logits']

        if torch.cuda.is_available() and device is not None:
            add_feats = add_feats.to(device)
            batch_node_feats = batch_node_feats.to(device)
            #add_feats = add_feats.cuda()
            #batch_node_feats = batch_node_feats.cuda()

        # LayerNorm on minibatch of per-residue protein embeddings, per-node embeddings
        add_feats = self.prot_norm(add_feats)
        batch_node_feats = self.mol_norm(batch_node_feats)

        prot_attention_maps, mol_attention_maps = self.cross_attn.gen_attn_maps(add_feats, batch_node_feats, seq_mask, node_mask)

        return prot_attention_maps, mol_attention_maps



class Mol_JointPredictor(nn.Module):
    """GCN-based model for regression and classification on graphs with cross-attention layer
    before classification of ORs by MLP head. Jointly predicts percepts.

    GCN is introduced in `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__. This model is based on GCN and can be used
    for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers. By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If None, no activation will be applied. If not None, ``activation[i]`` gives the
        activation function to be used for the i-th GCN layer. ``len(activation)`` equals
        the number of GCN layers. By default, ReLU is applied for all GCN layers.
    add_feats : int 
        Number of input OR features.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 add_feats = None, prot_feats = 1280, max_seq_len = 705, max_node_len = 22,
                 mol2_prot = False,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(Mol_JointPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        self.cross_attn = CrossAttention(prot_feats, gnn_out_feats, mol2prot = mol2_prot)
        
        self.predict_OR = MLPPredictor(max_seq_len + max_node_len, predictor_hidden_feats, 
                                    n_tasks[0], predictor_dropout)
        ## Percept prediction head
        self.predict_scent = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks[1], predictor_dropout)

        """
        if add_feats:
            self.predict = MLPPredictor(2 * gnn_out_feats + add_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)
        else:
            self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)
        """
    def forward(self, bg, feats, add_feats = None, seq_mask = None, node_mask = None):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization
        add_feats : FloatTensor of shape (B, n_residues, emb_dim)
            * B for the number of graph-OR pairs in the batch
            * n_residues is the number of residues in the protein sequence (padded to max_seq_len)
            * emb_dim is the embedding dimension of the protein sequence
        seq_mask : FloatTensor of shape (B, n_residues)
        node_mask : FloatTensor of shape (B, N)
        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        #print(bg)
        #print(feats)
        node_feats = self.gnn(bg, feats) ## problem causing NaNs is here
        graph_feats = self.readout(bg, node_feats)
        #print('node feats')
        #print(node_feats)
        if add_feats is not None:
            ## feed logits into bg for us to unbatch and index
            ## into correct graphs
            #bg.ndata['logits'] = node_feats
            
            ## Unbatch a batched graph
            graphs = dgl.unbatch(bg)
            batch_node_feats = torch.zeros((len(graphs), 22, node_feats.shape[1]))
            
            ## fill in batch_node feats with node feats from each graph
            counter = 0
            for i in range(len(graphs)):
                n_nodes = graphs[i].num_nodes()
                batch_node_feats[i][:n_nodes] = node_feats[counter:n_nodes + counter]
                counter+=n_nodes
                #batch_node_feats[i][:graphs[i].num_nodes()] = graphs[i].ndata['logits']
            
            #print('post readout graph feats')
            #print(graph_feats)
            ## Concatenate OR features to graph_feats before prediction
            ## outputs a tensor of shape (B, max_seq_len)
            
            ## Now, batch_node_feats is a padded batch of 2d tensors of node-level features.
            ## We pad the sequence tensor in the cross_attn forward pass
            
            ## set add_feats, batch_node_feats to cuda
            if torch.cuda.is_available():
                add_feats = add_feats.cuda()
                batch_node_feats = batch_node_feats.cuda()
            
            #print("ADD FEATS:") 
            #print(add_feats)
            #print("BATCH NODE FEATS:")
            #print(batch_node_feats)
            
            OR_feats = self.cross_attn(add_feats, batch_node_feats, seq_mask, node_mask)
        #if add_feats is not None:
        #    graph_feats = torch.cat((graph_feats, add_feats), dim=1)
        if add_feats is None:
            return self.predict_scent(graph_feats)
        else:
            return self.predict_scent(graph_feats), self.predict_OR(OR_feats)
