# Mapping the combinatorial coding between olfactory receptors and perception with deep learning

This repository contains the source code, plotting notebooks, and training data for the paper '[Mapping the combinatorial coding between olfactory receptors and perception with deep learning](https://www.biorxiv.org/content/10.1101/2024.09.16.613334v1)' (v2 in preparation).

Model weights, training data, and pre-generated HORDE / M2OR OR activation logits are available at:

**[Olfaction model weights and data (Zenodo)](https://zenodo.org/records/13765978)** | **[Google Drive mirror](https://drive.google.com/drive/folders/1cXUZsCH_2hmivVtdcyacZf_A8AZ_tmYN?usp=drive_link)**

The `checkpoints/` folder contains representative weights for the MolOR (odorant–receptor) and GCN (odorant–percept) models, including the upstream MolOR used to generate OR activation features for percept training and the MPNN-encoder variant. The `data/` folder contains canonical pre-generated OR activation logits (weighted/unweighted, HORDE/M2OR) and the null-distribution pickle backing the receptor-specificity analysis. Each subfolder includes its own README with provenance and usage notes.

Both bundles are distributed as `.tar.gz` archives. After downloading, extract them before moving the contents:

```bash
tar -xzf olfaction_data.tar.gz
tar -xzf olfaction_checkpoints.tar.gz
```

Then place the extracted `.pt` files under `data/datasets/` to run the percept ablations.

For an example of running inference with the MolOR model over the HORDE set of receptor sequences (including pseudogene controls), refer to `scripts/generate_OR_predictions_pseudogenes.py`.

## Environment

```bash
conda env create -f olfaction.yml
conda activate olfaction
```

## Files of note

### Training entry points
- `classification_ESM.py`: trains odorant-receptor models (MolOR) with fused per-residue ESM embeddings and bidirectional cross-attention. The `--model_encoder` flag selects between GCN and MPNN molecular encoders; configs live under `data/configures/M2OR_Pairs/` (e.g. `MolOR_canonical.json`, `MolOR_MPNN_canonical.json`). Requires ESM embeddings pre-computed on disk; first run will cache them.
- `classification_OR_feat_ESM.py`: trains odorant-percept models using predicted MolOR activations as input features (alongside the molecular GCN). Requires OR activation logits pre-computed on disk, or will run inference first to generate them for the given dataset.
- `classification.py`: basic GCN/MPNN classification baselines without ESM features.

### Ablation and reproduction scripts (under `scripts/`)
- `run_OR_percept_ablations_HORDE.sh`: main paper ablation — scales # of HORDE OR activations as input features for odorant-percept prediction. After downloading data from Zenodo into `data/datasets/`, run `bash scripts/run_OR_percept_ablations_HORDE.sh`.
- `run_OR_percept_ablations.sh`: equivalent ablation against the M2OR receptor set (1237 ORs).
- `run_OR_percept_ablations_all_DBs.sh`: ablation over the union of HORDE and M2OR ORs.
- `generate_OR_predictions_pseudogenes.py`: generates MolOR activation logits for HORDE receptors (functional and pseudogene splits).
- `prepare_enzpred_data.py`: produces the M2OR train/val/test splits used for the Goldman et al. (FFN+ESM) and PerceiverCPI baselines.
- `blast_uniprot.py`, `get_gene_uniprot_IDs_blast.py`, `merge_blast_annotations.py`, `m2or_ed_distance_matrix.py`, `get_HORDE_metadata.ipynb`: receptor annotation and pre-processing utilities.

### Analysis notebooks (under `notebooks/`)
- `fig2_plots.ipynb`, `figures_OR_percept.ipynb`, `percept_OR_plots.ipynb`: main-text figures.
- `fig4_stat_tests.ipynb`: statistical analyses including Benjamini–Hochberg-corrected ablation comparisons and the Jonckheere–Terpstra trend test reported in Table S1.
- `nutty_receptor_analysis.ipynb`, `filtered_nutty_receptor_analysis.ipynb`, `OR_subfamily_analysis.ipynb`, `cross_task_stats.ipynb`, `percept_receptor_null_distribution.ipynb`: per-percept and per-receptor analyses.
- `test_OR_logits_shuffle.ipynb`: shuffled-OR-logits control referenced in the revisions.

### Receptor binding pre-processing (under `receptor_binding/`)
Notebooks and utilities for preparing the M2OR pairwise dataset and computing receptor-level statistics (sequence-similarity matrix, BLAST-based annotations).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
