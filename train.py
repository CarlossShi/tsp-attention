import pickle
import tqdm
import time
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import get_device, get_costs
from model import AttentionModel

d_model = 128  # 128
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 1
dim_feedforward = 512  # 512
num_node_features = 2  # consistent with [pyg](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#)

seed = 1234
graph_size = 20
num_batches = 1000  # 2500
num_epochs = 100  # 100
batch_size = 512  # 512
sample_size = graph_size  # use for calculating baseline, assign a starting vertex for each sample in forward
num_instances = batch_size // sample_size
lr = 1e-4

torch.manual_seed(seed=seed)
device, device_count = get_device()
model = AttentionModel(
    d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward, dropout=0, activation=F.relu,
    norm_parameters={'method': 'batch', 'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
    norm_first=False, device=device,
    num_node_features=num_node_features
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
run_name = time.strftime("%Y%m%d-%H%M%S")
os.makedirs('checkpoints/{}'.format(run_name), exist_ok=True)
os.makedirs('runs/{}'.format(run_name), exist_ok=True)
writer = SummaryWriter('runs/{}'.format(run_name))
with open('data/{}_test_seed1234.pkl'.format(graph_size), 'rb') as f:
    data = torch.tensor(pickle.load(f)[:batch_size]).to(device)  # (batch_size, graph_size, num_node_features)

for epoch in tqdm.tqdm(range(num_epochs), desc='outer loop', position=0):
    model.train()
    for batch in tqdm.tqdm(range(num_batches), desc='inner loop', position=1, leave=False):
        x = torch.rand(num_instances, graph_size, num_node_features, device=device).repeat_interleave(sample_size, dim=0)  # (batch_size, graph_size, num_node_features)
        start_vertices = torch.arange(sample_size, device=device).repeat(num_instances)  # (batch_size)
        tours, log_prob_sums = model(x, start_vertices, greedy=False)
        costs = get_costs(x, tours)  # (batch_size)
        costs_baseline = torch.mean(costs.view(num_instances, sample_size), dim=1)  # (num_instances)
        costs_min = torch.min(costs.view(num_instances, sample_size), dim=1)[0]  # (num_instances)
        loss = torch.mean((costs - costs_baseline.repeat_interleave(sample_size)) * log_prob_sums)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        tours, log_prob_sums = model(data, torch.tensor([0] * batch_size, device=device), greedy=True)
        validation_cost = torch.mean(get_costs(data, tours)).item()
    print('training loss: {:.2f}, validation cost: {:.2f}'.format(
        loss.item(), validation_cost)
    )
    writer.add_scalar('training loss', loss.item(), epoch)
    writer.add_scalar('validation cost', validation_cost, epoch)
    torch.save({
        'epoch': epoch,
        'model.state_dict': model.state_dict(),
        'optimizer.state_dict': optimizer.state_dict(),
        'training loss': loss.item(),
        'validation cost': validation_cost,
    }, 'checkpoints/{}/{}.pt'.format(run_name, epoch))
