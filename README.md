# Mapping the combinatorial coding between olfactory receptors and perception with deep learning

This repository contains the source code, plotting notebooks, and training data for the paper '[Mapping the combinatorial coding between olfactory receptors and perception with deep learning](https://www.biorxiv.org/content/10.1101/2024.09.16.613334v1)'. This repository is actively in development, and we will add more instructions on training your own olfactory models on custom datasets, running inference, and generating activation maps for novel odorants.

We will add a Zenodo link shortly after preprint, containing model weights for the odorant-receptor and odorant-percept models. We will also include the OR logits for both the HORDE and M2OR receptor datasets, to run percept prediction with the OR activations as supplementary features. We are also working to add a custom dataloader for running inference with both models shortly, so users can score larger datasets of molecules for potential receptor and percept codes. For an example of doing so with the MolOR model on the HORDE set of receptor sequences, refer to `scripts/generate_OR_predictions_pseudogenes.py`.

Files of note:
-  `classification_ESM.py`: code for training odorant-receptor models, using fused per-residue ESM embeddings. Requires ESM embeddings pre-computed on disk.
- `classification_OR_feat_ESM.py`: code for training odorant-percept models, using predicted activations from MolOR. Requires OR activation logits pre-computed on disk, or will run inference first to generate for given dataset.
- `scripts/run_OR_percept_ablations_HORDE.sh`: script to reproduce main ablation in paper, scaling # of OR activations from HORDE dataset for odorant percept prediction (after downloading data from zenodo and dummping in `data/datasets`, simply run `bash run_OR_percept_ablations_HORDE.sh`).

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
