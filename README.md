# Role of Momentum in Smoothing Objective Function and Generalizability of Deep Neural Networks
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for image classification.  
shb.py used the implementation of "Understanding the Role of Momentum in Stochastic Gradient Methods" (NeurIPS2019).   
See <<https://github.com/Kipok/understanding-momentum>>.

# Abstract
For nonconvex objective functions, including deep neural networks, stochastic gradient descent (SGD) with momentum has fast convergence and excellent generalizability, but a theoretical explanation for this is lacking. In contrast to previous studies that defined the stochastic noise that occurs during optimization as the variance of the stochastic gradient, we define it as the gap between the search direction of the optimizer and the steepest descent direction and show that its level dominates generalizability of the model. We also show that the stochastic noise in SGD with momentum smoothes the objective function, the degree of which is determined by the learning rate, the batch size, the momentum factor, the variance of the stochastic gradient, and the upper bound of the gradient norm. By numerically deriving the stochastic noise level in SGD and SGD with momentum, we provide theoretical findings that help explain the training dynamics of SGD with momentum, which were not explained by previous studies on convergence and stability. We also provide experimental results supporting our assertion that model generalizability depends on the stochastic noise level.

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
