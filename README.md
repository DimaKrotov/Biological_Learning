# Biological_Learning
AMATH 534 final project on "biological" learning for MNIST based on the paper [Unsupervised Learning by Competing Hidden Units](https://doi.org/10.1073/pnas.1820458116) by D.Krotov and J.Hopfield.

## Project structure 
* `models/`: contains the neural network models and evaluation metric
* `trainer/`: contains the training & testing loop of the model
* `cmd.py`: command line arguments helper
* `main.py`: main training file logic
* `Unsupervised_learning_algorithm_MNIST.ipynb`: the implementation of the biological learning rules from the original author

## Training models
At the top of the directory, run 
```python
python main.py
```
This will create (if not already existed) the follow folders:
* `logs`: contains the training hyperparameters, tensorboard logs, and saved model checkpoints
* `data`: the default location for PyTorch to download the MNIST and CIFAR 10 datasets

To see all currently available command line arguments, run 
```python
python main.py -h
```

## Visualization
Run
```
tensorboard --logdir logs
```
