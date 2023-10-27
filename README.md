# Dynamical Graph Convolutional Neural Network
The **DGCNN** is a specific graph neural network architecture that leverages dynamic learning to extract patterns from graph-structured data that can dynamically change over time.

## Modules versions
* tensorboard == 2.13.0
* tensorflow == 2.13.0
* torch == 1.12.1
* torch-geometric == 2.3.1
* torch-scatter == 2.1.0+pt112cu102
* torchaudio == 0.12.1
* torcheeg == 1.0.11

## Run code
```
python script_name.py num_epochs ./train_fold
```
For example:
```
python dgcnn.py 100 my_first_train
```

