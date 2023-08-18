from ogb.utils import smiles2graph
import pandas as pd

def smiles_to_graph_obj(smiles):
    graph_objs = []
    for smile in smiles:
        graph_objs.append(smiles2graph(smile))
    
    return graph_objs

def get_graphormer_df(input_df):
    """Takes multi-task dataset and converts to format for Huggingface dataset needed to train graphormer.

    Args:
        input_df (pandas.Dataframe): dataframe where first column is smiles, and all following columns are targets
    """
    
    smiles = input_df['smiles'].tolist()
    
    ## initilaize empty list of dictionaries with keys 'edge_index', 'edge_attr',  'y', 'num_nodes', 'node_feat'
    graph_dict = []
    
    graph_objs = smiles_to_graph_obj(smiles)
    
    #graphormer_df = pd.DataFrame(columns=['edge_index', 'node_feat', 'edge_attr' 'num_nodes', 'y'], index=range(len(smiles)))
    
    for i, graph_obj in enumerate(graph_objs):
        graph_obj['edge_index'] = graph_obj['edge_index'].tolist()
        graph_obj['edge_feat'] = graph_obj['edge_feat'].tolist()
        graph_obj['edge_attr'] = graph_obj['edge_feat']
        ## renive edge_feat from graph_obj
        graph_obj.pop('edge_feat')
        if not input_df.iloc[i, 1:].isnull().values.any():
            graph_obj['y'] = input_df.iloc[i, 1:].astype(int).tolist()
        else:
            y_lst = input_df.iloc[i, 1:].tolist()
            for i, y in enumerate(y_lst):
                if type(y_lst[i]) == 'int64' or type(y_lst[i]) == float:
                    y_lst[i] = int(y_lst[i])
            graph_obj['y'] = y_lst
        graph_obj['node_feat'] = graph_obj['node_feat'].tolist()
        
        graph_dict.append({'edge_index' : graph_obj['edge_index'], 'edge_attr' : graph_obj['edge_attr'], 'y': graph_obj['y'], 'num_nodes' : graph_obj['num_nodes'], 'node_feat' : graph_obj['node_feat']})
        """
        print(graph_obj)
        print(smiles[i])
        graphormer_df.at[i, 'edge_index'] = graph_obj['edge_index']
        graphormer_df.at[i, 'node_feat'] = graph_obj['node_feat']
        
        ## Convert graph_obj['edge_feat'] to list of lists
        edge_feats = graph_obj['edge_feat'].tolist()
        print (edge_feats)
        graphormer_df.at[i, 'edge_attr'] = [edge_feat for edge_feat in edge_feats]  ## OGB uses 'edge_feat' as key
        #graphormer_df.loc[i, 'edge_attr'] = graph_obj['edge_feat']
        graphormer_df.at[i, 'num_nodes'] = graph_obj['num_nodes']
        graphormer_df.at[i, 'y'] = input_df.iloc[i, 1:].tolist()    
        
    return graphormer_df
    """
    return graph_dict
    #return graph_objs
    print (graph_objs)