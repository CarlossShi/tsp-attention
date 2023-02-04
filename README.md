# tsp-attention
 Attention model for solving Traveling Salesman Problem (TSP), implemented by Pytorch with Policy Optimization with Multiple Optima (POMO) baseline.

# Requirments

- python 3.9.16
- pytorch 1.13.1
- pytorch-cuda 11.7

# Getting Started

Run `generate.py` and then `train.py`.

# References

- Algorithm from [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475) and `get_costs` from [attention-learn-to-route/problems/tsp/problem_tsp.py](https://github.com/wouterkool/attention-learn-to-route/blob/6dbad47a415a87b5048df8802a74081a193fe240/problems/tsp/problem_tsp.py#L14)
- POMO baseline idea from [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://arxiv.org/abs/2010.16011)
- Code style from [The Transformer Network for the Traveling Salesman Problem](https://arxiv.org/pdf/2103.03012.pdf) and [xbresson/TSP_Transformer](https://github.com/xbresson/TSP_Transformer)
- Transformer model from [torch.nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
