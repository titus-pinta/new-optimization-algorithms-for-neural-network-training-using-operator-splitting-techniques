# NEW OPTIMIZATION ALGORITHMS FOR NEURAL NETWORK TRAINING USING OPERATOR SPLITTING TECHNIQUES
## Description
This is the GitHub repository for the article [ NEW OPTIMIZATION ALGORITHMS FOR NEURAL NETWORK TRAINING USING OPERATOR SPLITTING TECHNIQUES](/google.com)
## Abstract
In the following paper we present a new type of optimization algorithms adapted for neural network training. These algorithms are based upon sequential operator splitting technique for some associated dynamical systems. Furthermore, we investigate through numerical simulations the empirical rate of convergence of these iterative schemes toward a local minimum of the loss function, with some suitable choices of the underlying hyper-parameters. We validate the convergence of these optimizers using the results of the accuracy and of the loss function on the MNIST, MNIST-Fashion and CIFAR 10 classification datasets.
## Requirements
+ pytorch 
+ numpy
+ matplotlib
+ seaborn
+ dill

pip install -r requirements.txt
## Help
```bash
python main.py --help
```

```bash
python main.py --optimhelp
```

```bash
python main.py --losshelp
```

## Usage

```bash
python main.py --stoch --optim SSA1 --lr 0.1
```
runs ssa1 on the MNIST dataset, computing the gradient and taking a step for each mini batch


## Replicate Results

## View Saved Results

