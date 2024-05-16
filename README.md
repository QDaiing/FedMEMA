# Federated Modality-specific Encoders and Multimodal Anchors for Personalized Brain Tumor Segmentation

### Introduction

This repository is for our AAAI2024 paper '[Federated Modality-specific Encoders and Multimodal Anchors for Personalized Brain Tumor Segmentation]'.

cl_train.py - training code for clients

cl_train_glb.py - training code for server

fl_train_clsPasData_async.py - training code for federated learning (our method)

### Data Preparation
Get Brats2020 dataset [here](https://www.med.upenn.edu/cbica/brats2020/data.html) and refer to [RFNet](https://github.com/dyh127/RFNet).

### Usage
1. Clone the repository:

   ```shell
   git clone https://github.com/QDaiing/FedMEMA.git

   cd FedMEMA
   ```
   
2. Set hyperparameters via options.py and
   Train the model:
 
   ```shell
   bash run.sh
   ```
