# GWDMTI

## Overview

MicroRNA (miRNA) interactions with messenger RNA (mRNA) are essential for various biological processes, and accurately predicting these interactions is crucial for understanding their mechanisms. Traditional experimental methods often face limitations, making it increasingly important to develop robust predictive models for identifying potential miRNA targets. Current methods tend to rely solely on potential miRNA target sites and do not fully utilize the entire mRNA sequence, which can lead to a loss of crucial features.

To address these limitations, we introduce GWDMTI, a novel deep learning model designed to enhance the prediction of miRNA-target mRNA interactions. GWDMTI leverages both node and sequence features of miRNA and mRNA, aiming to improve predictive performance by overcoming the shortcomings of existing methods.

## Methodology

- **Feature Extraction**: We utilize RNA2vec to train on RNA data, obtaining RNA word vector representations.
- **Sequence Feature Mining**: Convolutional Neural Networks (CNN) and Bidirectional Gated Recurrent Units (BiGRU) are employed to extract RNA sequence features.
- **Node Features**: GraRep is used to derive node features.
- **Feature Integration**: A Deep Neural Network (DNN) integrates sequence and node features to provide a comprehensive prediction of miRNA-mRNA interactions.

## Performance

The GWDMTI model has demonstrated robust performance on the MTIS-9214 dataset, achieving:
- **Accuracy**: 85.892%
- **AUC (Area Under Curve)**: 0.9389
- **AUPR (Area Under Precision-Recall Curve)**: 0.9392

The model also shows high cross-dataset consistency, highlighting its notable referential value for advancing the study of miRNA-target mRNA interactions and indicating its utility and relevance in the field.

## Installation

Ensure you have Python 3.9 installed.
- **Software Versions**

Key dependencies with version specifications:

```bash
- PyTorch == 2.0.1
- TensorFlow == 2.14.0
- Transformers == 4.41.2
- Keras == 2.14.0
- PyTorch Geometric == 2.5.3
- CUDA Toolkit == 11.8
- NumPy == 1.23.5
- SciPy == 1.10.1
- Pandas == 2.0.3
- Scikit-learn == 1.3.0
- XGBoost == 2.0.3
- LightGBM == 4.4.0
- SpaCy == 3.0.0
- Tokenizers == 0.19.1
- Gensim == 4.3.3
- Jieba == 0.42.1
- Biopython == 1.83
- Biomart == 0.9.2
- Matplotlib == 3.4.3
- Seaborn == 0.13.2
- Plotly == 5.9.0
