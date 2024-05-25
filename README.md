# Role of Momentum in Smoothing Objective Function in Implicit Graduated Optimization
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for image classification.  
shb.py used the implementation of "Understanding the Role of Momentum in Stochastic Gradient Methods" (NeurIPS2019).   
See <<https://github.com/Kipok/understanding-momentum>>.

# Additional Experiments Results for Reviewers
Please check "Nbb-CIFAR-SHB.pdf".
Figure plots the SFO complexities for SHB with $\epsilon \in \lbrace 0.5, 0.7, 1.0 \rbrace$ needed to train ResNet18 on CIFAR100 dataset versus batch size.
The double circle symbol denotes the measured critical batch size that minimizes SFO complexity. The square symbol denotes the our theoretical lower bounds of critical batch size.

# Abstract
While stochastic gradient descent (SGD) with momentum has fast convergence and excellent generalizability, a theoretical explanation for this is lacking. In this paper, we show that SGD with momentum smooths the objective function, the degree of which is determined by the learning rate, the batch size, the momentum factor, the variance of the stochastic gradient, and the upper bound of the gradient norm. This theoretical finding reveals why momentum improves generalizability and provides new insights into the role of the hyperparameters, including momentum factor. We also present an implicit graduated optimization algorithm that exploits the smoothing properties of SGD with momentum and provide experimental results supporting our assertion that SGD with momentum smooths the objective function.

# Wandb Setup
Please change entity name `XXXXXX` to your wandb entitiy.
```
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help='entity of wandb team')
```

# Usage
Please select method.
```
parser.add_argument('--method', default="constant", type=str, help="constant, lr, beta, lr-batch, beta-batch, lr-beta, cosine, exp")
```

"constant" means constant learning rate, batch size, and momentum.  
"lr" means only learning rate decayed, with constant batch size and momentum.  
"beta" means only momnetum decayed, with constant learning rate and batch size.  
"lr-batch" means lr decayed and batch size increased, with constant momentum.  
"beta-batch" means momentum decayed and batch size increased, with constant learning rate.  
"lr-beta" means lr decayed and momentum decayed, with constant batch size.  
"cosine" means cosine annealing learning rate schedule, with constant batch size and momentum.  
"exp" means exponential decay learning rate schedule, with constant batch size and momentum.
