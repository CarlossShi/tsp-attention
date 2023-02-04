import os
import numpy as np
import pickle


seed = 1234
np.random.seed(seed=seed)
os.makedirs('data', exist_ok=True)
for graph_size in [20, 50, 100]:
    with open('data/{}_test_seed{}.pkl'.format(graph_size, seed), 'wb') as f:
        pickle.dump(np.random.uniform(size=(10000, graph_size, 2)).tolist(), f)
