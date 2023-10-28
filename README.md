# Dynamical Graph Convolutional Neural Network
The **DGCNN** is a specific graph neural network architecture that leverages dynamic learning to extract patterns from graph-structured data that can dynamically change over time.


## Dataset SEED-IV
The model has been trained using public dataset SEED-IV, in particular we used two features:
* Differencial entropy
* Power spectral density

## Modules versions
* ```tensorboard```
* ```tensorflow```
* ```torch```
* ```torch-geometric```
* ```torch-scatter```
* ```torchaudio```
* ```torcheeg```

## Run code
```
python script_name.py num_epochs ./train_fold
```
For example:
```
python dgcnn.py 100 my_first_train
```

## Run Tensorboard
```
tensorboard --logdir=/path/to/train_fold --bind_all --port=<number_of_the_port>
```

