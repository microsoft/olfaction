"""Prepare M2OR pairwise data in enz-pred format.

Reads the PerceiverCPI train/val/test CSVs and produces:
1. A combined CSV with columns (SEQ, SUBSTRATES, Activity) for enz-pred
2. A pickle file with train/val/test index arrays for the PresetSplitter
"""

import pandas as pd
import numpy as np
import pickle
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
OUT_CSV = os.path.join(os.path.expanduser("~"), "enz-pred", "data", "processed", "m2or_binary.csv")
OUT_PICKLE = os.path.join(os.path.expanduser("~"), "enz-pred", "data", "processed", "m2or_split_indices.p")

def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "perceiver_M2OR_train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "perceiver_M2OR_val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "perceiver_M2OR_test.csv"))

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    n_train = len(train)
    n_val = len(val)
    n_test = len(test)

    combined = pd.concat([train, val, test], ignore_index=True)
    combined = combined.rename(columns={
        "smiles": "SUBSTRATES",
        "sequence": "SEQ",
        "label": "Activity",
    })

    # Save combined CSV with numeric index (enz-pred expects index_col=0)
    combined.to_csv(OUT_CSV)
    print(f"Saved combined CSV ({len(combined)} rows) to {OUT_CSV}")

    # Save split indices
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(split_indices, f)
    print(f"Saved split indices to {OUT_PICKLE}")
    print(f"  train: {len(train_idx)} [{train_idx[0]}..{train_idx[-1]}]")
    print(f"  val:   {len(val_idx)} [{val_idx[0]}..{val_idx[-1]}]")
    print(f"  test:  {len(test_idx)} [{test_idx[0]}..{test_idx[-1]}]")

if __name__ == "__main__":
    main()
