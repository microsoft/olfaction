import pandas as pd
import umap

def merge_gs_lf(gs_path = 'goodscents/goodscents_molecule_percept.csv', lf_path = 'leffingwell/leffingwell_molecule_percept.csv', chemprop = False):
    """Merge Goodscents and Leffingwell into a single dataframe, filter/preprocess."""
    # Merge ground truth and label functions
    gs = pd.read_csv(gs_path)
    lf = pd.read_csv(lf_path)
    # change to Int64 to preserve NaNs w/o padding w/ 0s.
    gs.loc[:,'lovage':'tallow'] = gs.loc[:, 'lovage': 'tallow'].astype('Int64')
    lf.loc[:,'alcoholic':'woody'] = lf.loc[:, 'alcoholic': 'woody'].astype('Int64')
    gs_lf = pd.merge(gs, lf, on='IsomericSMILES', how='outer')
    gs_lf = gs_lf.rename(columns={'Stimulus_x': 'Stimulus'})
    gs_lf = gs_lf.drop(columns=['Stimulus_y'])
    gs_lf = gs_lf.drop(columns=['Descriptors'])
    gs_lf['IUPACName'] = gs_lf['IUPACName_x'].fillna(gs_lf['IUPACName_y'])
    gs_lf.drop(columns=['IUPACName_x', 'IUPACName_y'], inplace=True)
    gs_lf['name'] = gs_lf['name_x'].fillna(gs_lf['name_y'])
    gs_lf.drop(columns=['name_x', 'name_y'], inplace=True)
    for col in gs_lf.columns:
        if col.endswith('_x'):
            gs_lf[col[:-2]] = gs_lf[[col, col[:-2]+'_y']].max(axis=1)
            gs_lf = gs_lf.drop(columns=[col])
        elif col.endswith('_y'):
            gs_lf = gs_lf.drop(columns=[col])
    gs_lf_deduplicated = gs_lf.drop_duplicates(subset=['IsomericSMILES'])
    gs_lf_deduplicated = gs_lf_deduplicated[ ['IsomericSMILES'] + [ col for col in gs_lf_deduplicated.columns if col != 'IsomericSMILES' ] ]
    for col in gs_lf_deduplicated.columns:
        if col not in ['IsomericSMILES', 'name', 'Stimulus', 'IUPACName', 'MolecularWeight', 'CID']:
            gs_lf_deduplicated[col] = gs_lf_deduplicated[col].astype('Int64')
    
    gs_lf_deduplicated_drop_odors = gs_lf_deduplicated

    for col in gs_lf_deduplicated_drop_odors.columns:
        if col not in ['IsomericSMILES', 'name', 'Stimulus', 'IUPACName', 'MolecularWeight', 'CID']:
            ## Count the number of 1s in each column, if less than 10, drop the column
            if gs_lf_deduplicated_drop_odors[col].sum() < 30:
                gs_lf_deduplicated_drop_odors = gs_lf_deduplicated_drop_odors.drop(columns=[col])
    
    if chemprop:
        gs_lf_deduplicated_drop_odors.drop(columns=['Stimulus', 'CID', 'IUPACName', 'MolecularWeight', 'name'], inplace=True)    
    return gs_lf_deduplicated_drop_odors

def get_class_weights(num_pos_labels):
    total = num_pos_labels.sum()
    return {i: (total/num_pos_labels[i]) for i in range(len(num_pos_labels))}

def get_class_weights_from_df(df):
    ## Assume first column is SMILES
    return list(get_class_weights(df.sum(axis=0)[1:]).values())

def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title='', start_features = 0, end_features= 300):  
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data.iloc[:, start_features:end_features])
    umap.plot.points(fit, labels=data['odor'])

def vary_umap_params(data, n_neighbors=[15, 20, 25, 30], min_dist=[0.1, 0.2, 0.3, 0.4], n_components=[2, 3], metric=['euclidean', 'cosine'], start_features = 0, end_features= 300):
    ## Asumes label is stored as column 'odor'
    for n in n_neighbors:
        for m in min_dist:
            for c in n_components:
                for met in metric:
                    draw_umap(data, n_neighbors=n, min_dist=m, n_components=c, metric=met, start_features=start_features, end_features=end_features)