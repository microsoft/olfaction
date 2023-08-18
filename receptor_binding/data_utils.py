from typing import List
from io import StringIO
from urllib import parse
from urllib.request import Request, urlopen
import requests
import pandas
import pandas as pd


class MutationError(Exception):
    pass

class UniprotNotFoundError(Exception):
    pass

class UniprotMultipleOutputError(Exception):
    pass

def get_uniprot_sequences(uniprot_ids: List, check_consistency: bool = True) -> pandas.DataFrame:
        """
        Retrieve uniprot sequences based on a list of uniprot sequence identifier.

        For large lists it is recommended to perform batch retrieval.

        Parameters:
        ----------
        uniprot_ids: list 
            list of uniprot identifier

        Returns:
        --------
        df : pandas.DataFrame
            pandas dataframe with uniprot id column, sequence column and query column.
        """
        base_url = "https://www.uniprot.org/uniprot/"
        data = []

        for uniprot_id in uniprot_ids:
            response = requests.get(f"{base_url}{uniprot_id}.fasta")
            if response.status_code == 200:
                fasta_data = response.text.split("\n>")
                
                for fasta_entry in fasta_data:
                    if len(fasta_entry) > 0 : 
                        header, sequence = fasta_entry.split("\n", 1)
                    else:
                        print('nothing here')
                    query = header.split("|")[1]
                    data.append({"Entry": uniprot_id, "Uniprot_Sequence": sequence.replace("\n", ""), "Query": query})
            else:
                print(f"Error retrieving sequences, status code: {response.status_code}")
        
        df_fasta = pandas.DataFrame(data)
        # it might happen that 2 different ids for a single query id are returned, split these rows
        df_fasta = df_fasta.assign(Query=df_fasta['Query'].str.split(',')).explode('Query')
        if check_consistency:
            set_uniprot_ids = set(uniprot_ids)
            set_output_ids = set(df_fasta['Entry'])
            intersect = set_output_ids.intersection(set_uniprot_ids)
            if len(intersect) < len(set_uniprot_ids):
                raise UniprotNotFoundError('Some uniprot IDs were not found: {}'.format(set_uniprot_ids.difference(set_output_ids)))
            elif len(intersect) > len(set_uniprot_ids):
                raise UniprotMultipleOutputError('More uniprot IDs found than inputs. Difference: {}'.format(set_output_ids.difference(set_uniprot_ids)))
        return df_fasta


def generate_pairwise_activity_df(df):
    """
    
    Generate pairwise activity df, with each unique canonical SMILES as a row, and each unique OR sequence as a column (there must be some overlap among sequences, need to condense).

    Parameters:
    ----------
    df: pandas.DataFrame 
        pandas dataframe with Sequence column, canonicalSMILES column and Responsive column, with no NaN values in the three columns.
    Returns:
    --------
    OR_odorant_df : pandas.DataFrame
        pandas dataframe with each unique canonical SMILES as a row, and each unique OR sequence as a column (there must be some overlap among sequences, need to condense).
    """
    unique_SMILES = df['canonicalSMILES'].unique()
    unique_seq = df['Sequence'].unique()
    OR_odorant_df = pd.DataFrame(columns = unique_seq, index = unique_SMILES)
    idx = 0

    for i in OR_odorant_df.index:
        #Iterate over columns of OR_odorant_df
        for j in OR_odorant_df.columns:
            pairwise_df = df[(df['Sequence'] == j) & (df['canonicalSMILES'] == i)]
            if pairwise_df.shape[0] == 0:
                ##OR_odorant_df.at[i, j] = None
                continue
            elif pairwise_df.shape[0] == 1:
                OR_odorant_df.at[i, j] = pairwise_df['Responsive'].values[0].astype('int')
            else:
                ## TODO: right now I take the max (to get as many active labels as possible), but should weigh labels based on experiment type.
                OR_odorant_df.at[i, j] = pairwise_df['Responsive'].values.max().astype('int')

        if idx % 100 == 0:
            print(idx)      
        idx+=1
    return OR_odorant_df

def generate_pairwise_activity_custom_df(df, label_col = 'Uniprot ID'):
    """
    
    Generate pairwis activity dataframe, where each row is a canonical SMILES and each
    column is a unique identifier specified by the parameter passed into 'label_col' (ex - Uniprot ID)
    
    Parameters:
    ----------
    df: pandas.DataFrame 
        pandas dataframe with Sequence column, canonicalSMILES column and Responsive column, with no NaN values in the three columns.
    Returns:
    --------
    OR_odorant_df : pandas.DataFrame
        pandas dataframe with each unique canonical SMILES as a row, and each unique OR sequence as a column (there must be some overlap among sequences, need to condense).
    """
    unique_SMILES = df['canonicalSMILES'].unique()
    unique_col = df[label_col].unique()
    OR_odorant_df = pd.DataFrame(columns = unique_col, index = unique_SMILES)
    idx = 0

    for i in OR_odorant_df.index:
        #Iterate over columns of OR_odorant_df
        for j in OR_odorant_df.columns:
            pairwise_df = df[(df[label_col] == j) & (df['canonicalSMILES'] == i)]
            if pairwise_df.shape[0] == 0:
                ##OR_odorant_df.at[i, j] = None
                continue
            elif pairwise_df.shape[0] == 1:
                OR_odorant_df.at[i, j] = pairwise_df['Responsive'].values[0].astype('int')
            else:
                ## TODO: right now I take the max (to get as many active labels as possible), but should weigh labels based on experiment type.
                OR_odorant_df.at[i, j] = pairwise_df['Responsive'].values.max().astype('int')

        if idx % 100 == 0:
            print(idx)
        idx+=1
    return OR_odorant_df


#m2or = pd.read_csv('M2OR/M2OR_2023_04_28_full_seq_mut_canonSMILES.csv')

## To generate collapsed pairwise df by Uniprot ID, run below lines
"""
m2or = pd.read_csv('M2OR/M2OR_2023_04_28_full_seq_mut_canonSMILES_Uniprot_ID.csv')

pairwise_activity_df = generate_pairwise_activity_custom_df(m2or)
pairwise_activity_df.to_csv('M2OR/M2OR_2023_04_28_pairwise_activity_Uniprot.csv')
pairwise_activity_df.to_parquet('M2OR/M2OR_2023_04_28_pairwise_activity_Uniprot.pq')
"""