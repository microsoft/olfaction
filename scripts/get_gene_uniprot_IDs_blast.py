#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLAST Sequence Augmentation Script

Exactly replicates the logic from blast_uniprot.py but for the seqs.csv format.
"""

from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description='Augment protein sequence CSV with UniProt and gene IDs via BLAST search')
    parser.add_argument('input_csv', help='Input CSV file containing protein sequences')
    parser.add_argument('output_csv', help='Output CSV file with augmented UniProt and gene IDs')
    parser.add_argument('--max-sequences', type=int, help='Maximum number of sequences to process (for testing)')
    
    args = parser.parse_args()
    
    # Read the CSV file (semicolon separated like seqs.csv)
    print(f"Reading {args.input_csv}...")
    df = pd.read_csv(args.input_csv, sep=';')
    print(f"Loaded {len(df)} rows")
    
    # Get unique sequences from _Sequence column (ignore mutations for finding OR classification)
    unique_sequences = df['_Sequence'].unique().tolist()
    print(f"Found {len(unique_sequences)} unique sequences")
    
    # Limit for testing if specified
    if args.max_sequences:
        unique_sequences = unique_sequences[:args.max_sequences]
        print(f"Limited to {args.max_sequences} sequences for testing")
    
    blast_results = []
    
    # Process each sequence exactly like the original script
    for sequence in unique_sequences:
        print(f"Processing sequence: {sequence[:50]}...")
        
        # Use EXACT same BLAST call as original script - note the space at beginning of entrez_query
        result_handle = NCBIWWW.qblast("blastp", "uniprot", sequence, entrez_query=" txid9606 [ORGN] OR txid9601 [ORGN] OR txid9598 [ORGN]")
        blast_record = NCBIXML.read(result_handle)

        if blast_record.alignments:
            hit = blast_record.alignments[0]
            query_id = hit.title.split()[0]  # This is the full "sp|Q8NGX5|O10K1_HUMAN" format
            uniprot_id = hit.accession       # This is just "Q8NGX5"
            print(f"  Found hit: {query_id} -> UniProt: {uniprot_id}")
            
            # Extract gene_id from query_id (format: sp|uniprot_id|gene_id)
            gene_id = None
            if '|' in query_id:
                parts = query_id.split('|')
                if len(parts) >= 3:
                    gene_id = parts[2]  # e.g., "O10K1_HUMAN"
            
            blast_results.append((query_id, uniprot_id, sequence, gene_id))
        else:
            print(f"  No alignments found")
            blast_results.append((None, None, sequence, None))

    # Create a lookup dictionary keyed by sequence
    blast_lookup = {}
    for query_id, uniprot_id, sequence, gene_id in blast_results:
        blast_lookup[sequence] = {'uniprot_id': uniprot_id, 'gene_id': gene_id, 'query_id': query_id}
    
    # Add new columns to dataframe
    df['blast_uniprot_id'] = df['_Sequence'].map(lambda x: blast_lookup.get(x, {}).get('uniprot_id'))
    df['blast_gene_id'] = df['_Sequence'].map(lambda x: blast_lookup.get(x, {}).get('gene_id'))
    df['blast_query_id'] = df['_Sequence'].map(lambda x: blast_lookup.get(x, {}).get('query_id'))
    
    # Save results exactly like original script (but as semicolon-separated)
    print(f"Saving results to {args.output_csv}...")
    df.to_csv(args.output_csv, sep=';', index=False)
    
    # Print summary
    total_found = df['blast_uniprot_id'].notna().sum()
    print(f"\nSummary:")
    print(f"Total sequences processed: {len(unique_sequences)}")
    print(f"UniProt IDs found: {total_found}")
    print(f"Gene IDs found: {df['blast_gene_id'].notna().sum()}")


if __name__ == '__main__':
    main()
