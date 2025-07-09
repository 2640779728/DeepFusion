# DeepFusion: Sequence Feature Embedding and Multi-modal Fusion for Cell-type-specific Prediction

## Overview

**DeepFusion** is a flexible deep learning pipeline for extracting, embedding, and fusing DNA sequence features for cell-type-specific prediction tasks. It supports both mouse (mESC) and human cell types by using different feature fusion models according to the biological context.

---

## Features

- **Automated k-mer embedding** (4-mer/5-mer/6-mer) for DNA sequences.
- **Flexible feature fusion:** Supports both 3-feature and 1-feature fusion for mouse and human data respectively.
- **Robust training scripts** with data balancing (SMOTE), stratified cross-validation, and detailed logging.
- **End-to-end workflow:** From bigWig/CSV input, through FASTA and embedding, to final model training and evaluation.
- **Ready for cell-type generalization and comparative analysis.**

---
Script Descriptions
DNasel.py:
Extracts numerical values from bigWig files based on regions specified in a CSV file and outputs them into a new CSV.

getfasta.py:
Converts a CSV file of genomic regions to a FASTA format file, extracting the correct DNA sequence from the genome FASTA file.

get_datavec.py:
Generates k-mer (4/5/6-mer) embedding vectors from DNA sequences using pre-trained word2vec models. Outputs vectors as CSV files (for each k).

LogUtils.py:
Logging utility for unified log formatting and file output.

model_2_fusion.py:
For mouse mESC and related cells: Two-branch neural network fusing 300-d k-mer sequence features and 3 additional features.

model_2_fusion2.py:
For most human cell types: Variant of the above; fuses 300-d sequence features and 1 additional feature.

train_specific_2_fusion.py:
Training script using model_2_fusion.py for mESC (mouse) and other cells with 3 features.

train_specific_2_fusion2.py:
Training script using model_2_fusion2.py for human cell types with 1 feature, supports ROC plotting.

testcell2.py:
Evaluation and prediction across multiple cell types, with all main performance metrics computed.

Typical Workflow

Extract features from bigWig:
```python
python DNasel.py
```

Generate FASTA from CSV:
python getfasta.py [cell_name] [genome]

Embed sequences using k-mer vectors:
python get_datavec.py [cell_name] [genome]

Test/evaluate models:
python testcell2.py

## **DeepSE demo**

For mESC/mouse (3 features):
python train_specific_2_fusion.py mESC 0.001 30

For human (1 feature):
python train_specific_2_fusion2.py spleen 0.001 30
