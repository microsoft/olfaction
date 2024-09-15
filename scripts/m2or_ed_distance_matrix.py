import seq_utils
import pandas as pd
import numpy as np

m2or = pd.read_csv("../receptor-binding/M2OR/M2OR_2023_04_28_full_seq_mut_canonSMILES.csv")

unique_sequences = m2or['Sequence'].unique().tolist()

#m2or_ed_matrix = seq_utils.generate_edit_distance_matrix(unique_sequences)
#m2or_ed_matrix_np = np.asarray(m2or_ed_matrix)
m2or_ed_matrix_np = seq_utils.generate_edlib_edit_distance_matrix(unique_sequences)

m2or_ed_matrix_norm = np.zeros((len(unique_sequences), len(unique_sequences))).astype(float)

for i in range(len(unique_sequences)):
    for j in range(len(unique_sequences)):
        m2or_ed_matrix_norm[i][j] = m2or_ed_matrix_np[i][j] / len(unique_sequences[i])

np.save("../receptor-binding/M2OR/M2OR_2023_04_28_normalized_edit_distance_matrix.npy", m2or_ed_matrix_norm)
