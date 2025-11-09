# Signature Verification System (Contrastive Siamese Network)

## Overview
This project implements a **deep learning-based offline signature verification system** using a **Siamese Neural Network** trained with **Contrastive Loss**.  
The model learns to distinguish between **genuine and forged signatures** by comparing pairs of images and computing their similarity in an embedding space.

---

## Features
- Signature pair generation for training (genuine vs forged)
- Siamese Network architecture with shared convolutional weights
  **Contrastive Loss** for distance-based verification
- Evaluation using **ROC Curve**, **Precision-Recall**, and **AUC**
- Automatic threshold detection for optimal classification
- Support for offline dataset loading (e.g., CEDAR, GPDS)
- Optional preprocessing for **noise reduction, seal removal, and binarization**

---

## Architecture
The system is composed of:
1. **Twin Convolutional Networks** (CNNs) that extract feature embeddings.
2. **Euclidean Distance Layer** to measure similarity.
3. **Contrastive Loss Function** that minimizes distance for genuine pairs and maximizes for forged pairs.

