# Given two lists of SMILES, ind_SMILES and canonical_SMILES, return a 2d array of the Tanimoto similarity between each pair of SMILES.
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem

def tanimoto_similarity(ind_SMILES, canonical_SMILES):
    tanimoto_sim = []
    for ind_smi in ind_SMILES:
        ind_mol = Chem.MolFromSmiles(ind_smi)
        if ind_mol is not None:
            ind_fp = FingerprintMols.FingerprintMol(ind_mol)
            tanimoto_sim.append([DataStructs.FingerprintSimilarity(ind_fp, FingerprintMols.FingerprintMol(Chem.MolFromSmiles(can_smi))) for can_smi in canonical_SMILES])
        else:
            tanimoto_sim.append([None for can_smi in canonical_SMILES])
    return tanimoto_sim

def generate_edlib_edit_distance_matrix(sequences):
    ## initialize empty 2d Numpy array with size (len(sequences), len(sequences))
    import numpy as np
    import edlib
    edlib_matrix = np.empty((len(sequences), len(sequences)), dtype=np.int32)
    
    for i, seq1 in enumerate(sequences):
        assert isinstance(seq1, str), "All sequences must be strings."
        for j, seq2 in enumerate(sequences):
            edlib_matrix[i][j] = edlib.align(seq1, seq2)['editDistance']
    
    return edlib_matrix

def generate_edit_distance_matrix(sequences):
    """Generate edit distance matrix of protein sequences with varying lengths.

    Args:
        sequences (List[str]): List of protein sequences

    Returns:
        List[List[int]]: Edit distance matrix
    """
    num_sequences = len(sequences)
    matrix = [[0] * num_sequences for _ in range(num_sequences)]

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            distance = calculate_edit_distance(sequences[i], sequences[j])
            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix

def calculate_edit_distance(seq1, seq2):
    """Calculate the Levenshtein distance between two protein sequences.

    Args:
        seq1 (str): First protein sequence
        seq2 (str): Second protein sequence

    Returns:
        int: Edit distance between the two sequences
    """
    m = len(seq1)
    n = len(seq2)

    if m == 0:
        return n
    if n == 0:
        return m

    # Create a matrix to store the edit distances
    matrix = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column of the matrix
    for i in range(m + 1):
        matrix[i][0] = i
    for j in range(n + 1):
        matrix[0][j] = j

    # Compute the edit distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,         # Deletion
                matrix[i][j - 1] + 1,         # Insertion
                matrix[i - 1][j - 1] + cost   # Substitution
            )

    return matrix[m][n]
