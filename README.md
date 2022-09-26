# Asset pricing via deep graph learning to incorporate heterogeneous predictors
This is the source code of our paper, 
named asset pricing via deep graph learning to incorporate heterogeneous predictors

## Model architechture
![image](./Pictures/model_architecture.png) 

## Overview
* `layers.py` contains two modules of the SA-GNN model: Matrix‐based feature fusion (`MatrixFusionLayer`) and 
  Self‐adaptive GNN (`ExplicitLayer`);
  
* `models.py` contains the SA-GNN model;

* `utils.py` contains data loading (`load_data`) 
  and evaluation metrics (`metrics`);

* `train_evaluation.py` puts all of the above together and is uesd to execute
a full training run on our dataset.






