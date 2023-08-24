This code is a part of the paper *Symbolic Relational Deep Reinforcement Learning based on Graph Neural Networks and Autoregressive Policy Decomposition*. 

# Brief instructions
Choose the experiment you want to run, and execute `python main.py`. Check `python main.py --help` for additional arguments. If you have a trained model, you can evaluate it with `python main.py -eval -load model.pt`. For each experiment we include some trained models in the `rrl-*/_store/` directory.

The results are logged into *stdout* and [wandb](https://github.com/wandb/client) experiment management system. Log-in with your account if necessary.

The code is based on [PyTorch](https://github.com/pytorch/pytorch) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

# Dependencies
The code requires at least Python 3.6.12.

Udpate pip with `pip install --upgrade pip`.
Install pytorch `pip install torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`.
Install pytorch geometric requirements:
`pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install scipy
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html`
(you can install gpu versions by replacing `+cpu` with `+cu101`)

Install other dependencies with `pip install -r requirements.txt`.

# How to train the specific models
## BlockWorld
`python main.py -boxes <N>`

## SysAdmin
`python main.py -nodes <N>` for SysAdmin-S
`python main.py -nodes <N> --multi` for SysAdmin-M

## Sokoban
`python main.py` for default unfiltered dataset (will download automatically)
`python main.py --custom <XxYxB>` for custom size environment of size XxY with B boxes

# Evaluate 
add `-eval -load <model.pt>` to the arguments

