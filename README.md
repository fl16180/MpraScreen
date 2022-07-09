# MPRA Screen Repository

This repository contains code supporting the data processing, model training, and analysis of the MpraNet model. This is a deep learning model trained to predict whether SNP sites have the kind of functional effect detected by MPRA assay. The model is trained on positive variants from existing MPRA experiments on the GM12878 cell line, with matched negatives sampled from the background genome. The model uses numerous features including epigenetic annotations and prior functional scores.

Features were obtained from ROADMAP, Eigen, and RegBase databases, and were processed with `setup_project_data.py`, `setup_neighbor_data.py`, and `setup_mixed_datasets.py`, which also generates the train and validation splits.

MPRANet was tuned with 5-fold CV using `search_hparam.py`, after which the final model was trained using `fit_mpra_models.py`

Subsequent analysis of the model was performed using evaluation scripts, such as `evaluate_results.py` and `sample_size_sim.py`.

We additionally provide an example script for using a trained MPRANet to predict on a user-given set of genomic variant locations in `predict_script.py`. 
