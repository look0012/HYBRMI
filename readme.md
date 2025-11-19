# HybRMI

## Overview

MicroRNAs (miRNAs) interact with messenger RNAs (mRNAs) to regulate key biological processes, with disruptions often linked to diseases such as cancer and neurodegenerative disorders. Predicting these interactions computationally is essential, as experimental validation is labor-intensive and limited in scale. However, many models prioritize binding sites over full sequence context and require manual feature engineering, risking the loss of critical information. We propose HybRMI, a hybrid GraRep-RNA2vec fusion framework that addresses these gaps by integrating RNA sequence embeddings with network topology for enhanced miRNA-mRNA interaction prediction. RNA2vec pre-trains vector representations from complete sequences, which are refined via convolutional neural networks and bidirectional gated recurrent units for deep feature extraction. GraRep complements this by embedding graph-based structural insights, with a deep neural network fusing the modalities for robust classification. Extending beyond core prediction, HybRMI's efficient design supports potential applications in consumer electronics, enabling personalized healthcare analytic through seamless analysis of distributed biomolecular data from devices like wearables and smartphones. This could empower real-time insights into gene regulation for individualized risk monitoring and interventions. On the benchmark dataset, HybRMI achieves 85.89% accuracy with 0.9389 AUC under five-fold cross-validation, outperforming benchmarks. Ablation experiments and case studies affirm the contributions of its modules. HybRMI thus advances bioinformatics tools while offering a pathway to intelligent consumer health systems. 


## Methodology

- **Feature Extraction**: We utilize RNA2vec to train on RNA data, obtaining RNA word vector representations.
- **Sequence Feature Mining**: Convolutional Neural Networks (CNN) and Bidirectional Gated Recurrent Units (BiGRU) are employed to extract RNA sequence features.
- **Node Features**: GraRep is used to derive node features.
- **Feature Integration**: A Deep Neural Network (DNN) integrates sequence and node features to provide a comprehensive prediction of miRNA-mRNA interactions.

## Performance

The HybRMI model has demonstrated robust performance on the MTIS-9214 dataset, achieving:
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

