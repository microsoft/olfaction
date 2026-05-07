#!/usr/bin/env python3
"""
Script to merge blast results with sequence data.
Adds uniprot_id and gene_id columns to seqs.csv based on matching sequences in blast_results.txt
"""

import pandas as pd
import re


def parse_blast_results(blast_file_path):
    """
    Parse blast_results.txt file to extract sequence-to-uniprot/gene mappings.
    Format: sequence\tsp|uniprot_id|gene_id_SPECIES
    """
    sequence_to_annotations = {}
    
    with open(blast_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    sequence = parts[0]
                    annotation = parts[1]
                    
                    # Parse annotation: sp|Q8NGX5|O10K1_HUMAN
                    match = re.match(r'sp\|([^|]+)\|([^_]+)_', annotation)
                    if match:
                        uniprot_id = match.group(1)
                        gene_id = match.group(2)
                        sequence_to_annotations[sequence] = {
                            'uniprot_id': uniprot_id,
                            'gene_id': gene_id
                        }
    
    return sequence_to_annotations


def merge_annotations_with_sequences(seqs_file_path, sequence_annotations, output_file_path):
    """
    Read seqs.csv, add uniprot_id and gene_id columns, and match sequences.
    """
    # Read the seqs.csv file
    df = pd.read_csv(seqs_file_path, sep=';')
    
    print(f"Read {len(df)} sequences from {seqs_file_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Add new columns
    df['uniprot_id'] = ''
    df['gene_id'] = ''
    
    # Match sequences and populate annotations
    matches_found = 0
    
    for idx, row in df.iterrows():
        sequence = row['_Sequence']
        if sequence in sequence_annotations:
            df.at[idx, 'uniprot_id'] = sequence_annotations[sequence]['uniprot_id']
            df.at[idx, 'gene_id'] = sequence_annotations[sequence]['gene_id']
            matches_found += 1
    
    print(f"Found matches for {matches_found} out of {len(df)} sequences")
    
    # Save the updated dataframe
    df.to_csv(output_file_path, sep=';', index=False)
    print(f"Saved updated file to {output_file_path}")
    
    return df


def main():
    blast_file = '/home/seyonec/olfaction/receptor_binding/blast_results.txt'
    seqs_file = '/home/seyonec/olfaction/data/datasets/M2OR/seqs.csv'
    output_file = '/home/seyonec/olfaction/data/datasets/M2OR/seqs_with_annotations.csv'
    
    print("Parsing blast results...")
    sequence_annotations = parse_blast_results(blast_file)
    print(f"Found {len(sequence_annotations)} sequence annotations")
    
    print("\nMerging with sequence data...")
    updated_df = merge_annotations_with_sequences(seqs_file, sequence_annotations, output_file)
    
    # Show some statistics
    non_empty_uniprot = (updated_df['uniprot_id'] != '').sum()
    non_empty_gene = (updated_df['gene_id'] != '').sum()
    
    print(f"\nSummary:")
    print(f"Total sequences: {len(updated_df)}")
    print(f"Sequences with uniprot_id: {non_empty_uniprot}")
    print(f"Sequences with gene_id: {non_empty_gene}")
    
    # Show a few examples of matched sequences
    matched_examples = updated_df[updated_df['uniprot_id'] != ''].head(5)
    if not matched_examples.empty:
        print(f"\nFirst 5 matched sequences:")
        for idx, row in matched_examples.iterrows():
            print(f"  seq_id: {row['seq_id']}, uniprot_id: {row['uniprot_id']}, gene_id: {row['gene_id']}")


if __name__ == "__main__":
    main()
