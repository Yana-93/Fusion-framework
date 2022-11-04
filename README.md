# Asset pricing via deep graph learning to incorporate heterogeneous predictors
This is the source code of our paper, 
named asset pricing via deep graph learning to incorporate heterogeneous predictors

## Model architechture
![image](./Pictures/model_architecture.png) 

## Overview
### Source code 
* `layers.py` contains two modules of the SA-GNN model: Matrix‐based feature fusion (`MatrixFusionLayer`) and 
  Self‐adaptive GNN (`ExplicitLayer`);
  
* `models.py` contains the SA-GNN model;

* `utils.py` contains data loading (`load_data`) 
  and evaluation metrics (`metrics`);

* `train_evaluation.py` puts all of the above together and is uesd to execute
a full training run on our dataset.

### Raw data & data preprocessing
Due to limited space, the raw data and corresponding data preprocessing code can be download from https://drive.google.com/drive/folders/1ss1bycGQ4l98YkvvOknSF6Bv_PM1MhS7?usp=sharing.

```
raw_data
├── raw_data_SP500
│   ├── DataProcessing
│   │   ├── Economic_Indicators_Data_Processing.ipynb
│   │   ├── Firm_Relations_Processing.ipynb
│   │   └── Textual_Media_Processing.ipynb
│   ├── EconomicData
│   ├── NewsData
│   └── RelationData
└── raw_data_SSE50
    ├── DataProcessing
    │   ├── Economic_Indicators_Data_Processing.ipynb
    │   ├── Firm_Relations_Processing.ipynb
    │   └── Textual_Media_Processing.ipynb
    ├── EconomicData
    ├── NewsData
    └── RelationData

```
Please note that due to news copyright issues, we only give partial news text data here.

## Contact
JiwenHuangFIC@gmail.com


