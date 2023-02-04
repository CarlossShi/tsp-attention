import torch


def get_device():
    print('torch.cuda.is_available():', torch.cuda.is_available())
    device_count = torch.cuda.device_count()
    print('torch.cuda.device_count():', device_count)
    device_idxes = list(range(device_count))
    print('device_idxes:', device_idxes)
    devices = [torch.cuda.device(_) for _ in device_idxes]
    print('devices:', devices)
    device_names = [torch.cuda.get_device_name(_) for _ in device_idxes]
    print('device_names:', device_names, '\n')

    current_device = torch.cuda.current_device()
    print('torch.cuda.current_device():', current_device)
    print('torch.cuda.device(current_device):', devices[current_device])
    print('torch.cuda.get_device_name(current_device):', device_names[current_device], '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, '\n')

    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    return device, device_count


def get_costs(x, tours):
    """
    # https://github.com/wouterkool/attention-learn-to-route/blob/6dbad47a415a87b5048df8802a74081a193fe240/problems/tsp/problem_tsp.py#L14
    @param x: (batch_size, graph_size, 2)
    @param tours: (batch_size, graph_size)
    @return: (batch_size)
    """
    # check that tours are valid, i.e. contain 0 to n -1
    assert (torch.arange(tours.size(1), out=tours.data.new()).view(1, -1).expand_as(tours) == tours.data.sort(1)[0]).all(), 'Invalid tour'
    # gather dataset in order of tour
    d = x.gather(1, tours.unsqueeze(-1).expand_as(x))
    # length is distance (L2-norm of difference) from each next location from its prev and of last from first
    return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)

