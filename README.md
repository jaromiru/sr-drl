This code is part of the paper *Symbolic Relational Deep Reinforcement Learning based on Graph Neural Networks*. 

Currently, only the BlockWorld experiment is available. Sokoban and SysAdmin are to be released.
Choose the experiment you want to run, and execute `python main.py`. Check `python main.py --help` for additional arguments. If you have a trained model, you can evaluate it with `python main.py -eval -load model.pt`. For each experiment we include some trained models in the `rrl-*/_store/` directory.

The results are logged into *stdout* and [wandb](https://github.com/wandb/client) experiment management system. Log-in with your account if necessary.

The code is based on [PyTorch](https://github.com/pytorch/pytorch) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).