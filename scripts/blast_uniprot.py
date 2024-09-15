#/home/t-seyonec/olfaction/receptor-binding/M2OR/M2OR_2023_04_28_unique_sequences.fasta

from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import pandas as pd

m2or = pd.read_csv("receptor-binding/M2OR/M2OR_2023_04_28_full_seq_mut_canonSMILES.csv")
m2or_nan = m2or[m2or['Uniprot ID'].isna()]
unique_sequences = m2or_nan['Sequence'].unique().tolist()

blast_results = []

for sequence in unique_sequences:
    result_handle = NCBIWWW.qblast("blastp", "uniprot", sequence, entrez_query=" txid9606 [ORGN] OR txid9601 [ORGN] OR txid9598 [ORGN]")
    blast_record = NCBIXML.read(result_handle)

    if blast_record.alignments:
        hit = blast_record.alignments[0]
        query_id = hit.title.split()[0]
        uniprot_id = hit.accession
        print(uniprot_id)
        blast_results.append((query_id, uniprot_id, sequence))

## Save results to a df with columns 'Sequence', 'Uniprot ID' and 'Query ID'
blast_results_df = pd.DataFrame(blast_results, columns=['Query ID', 'Uniprot ID', 'Sequence'])
blast_results_df.to_csv('receptor-binding/M2OR/M2OR_2023_04_28_blast_results.csv')