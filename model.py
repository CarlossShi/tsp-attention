from typing import Union, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.categorical import Categorical

norms_parameters = {
    'layer': {'method': 'layer', 'eps': 1e-5, 'momentum': None, 'affine': True, 'track_running_stats': None},
    'batch': {'method': 'batch', 'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
    'instance': {'method': 'instance', 'eps': 1e-5, 'momentum': 0.1, 'affine': False, 'track_running_stats': False}
}


class Norm(nn.Module):
    def __init__(self, num_features, method, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super(Norm, self).__init__()
        self.method = method
        if method == 'layer':
            assert momentum is None and track_running_stats is None, 'norm parameters and method are not matched'
            self.norm = nn.LayerNorm(num_features, eps, affine, device, dtype)
        elif method == 'batch':
            self.norm = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        elif method == 'instance':
            self.norm = nn.InstanceNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        else:
            assert False, 'unknown norm method: {}'.format(method)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x) if self.method == 'layer' else torch.transpose(self.norm(torch.transpose(x, 1, 2)), 1, 2)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 norm_parameters: dict = norms_parameters['layer'], batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = Norm(d_model, **(norm_parameters | factory_kwargs))
        self.norm2 = Norm(d_model, **(norm_parameters | factory_kwargs))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 norm_parameters: dict = norms_parameters['layer'], batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = Norm(d_model, **(norm_parameters | factory_kwargs))
        self.norm2 = Norm(d_model, **(norm_parameters | factory_kwargs))
        self.norm3 = Norm(d_model, **(norm_parameters | factory_kwargs))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class AttentionModel(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 3,
                 num_decoder_layers: int = 1, dim_feedforward: int = 512, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 norm_parameters: dict = norms_parameters['batch'], norm_first: bool = False, device=None, dtype=None,
                 num_node_features: int = 2) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AttentionModel, self).__init__()
        self.embedding = nn.Linear(num_node_features, d_model)
        batch_first = True
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, norm_parameters, batch_first, norm_first, **factory_kwargs
        )  # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )  # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html

        self.linear = nn.Linear(3 * d_model, d_model, **factory_kwargs)
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, norm_parameters, batch_first, norm_first, **factory_kwargs
        )  # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers
        )  # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=batch_first, **factory_kwargs
        )  # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, start_vertices, greedy=True):
        """

        @param x: (batch_size, graph_size, num_node_features)
        @param start_vertices: (batch_size), the first visited vertex of each batch
        @param greedy: bool
        @return: (batch_size, graph_size), (batch_size)
        """
        batch_size, graph_size, num_node_features = x.shape
        assert start_vertices.shape[0] == batch_size
        indexes = torch.arange(batch_size)
        h_vertices = self.encoder(self.embedding(x))  # (batch_size, graph_size, d_model)
        h_graph = torch.mean(h_vertices, dim=1)  # (batch_size, d_model)
        h_first = h_vertices[indexes, start_vertices]  # (batch_size, d_model)
        h_last = h_vertices[indexes, start_vertices]  # (batch_size, d_model)
        visited_mask = torch.zeros(batch_size, graph_size, dtype=torch.bool, device=x.device)  # (batch_size, graph_size)
        visited_mask[indexes, start_vertices] = True
        log_prob_list, action_list = [], [start_vertices]
        for t in range(graph_size - 1):
            assert all(torch.sum(visited_mask, dim=1) < graph_size), 'all vertices are masked in some instances'
            h_state = self.linear(torch.cat((h_graph, h_first, h_last), dim=1)).unsqueeze(1)  # (batch_size, 1, d_model)
            h = self.decoder(h_state, h_vertices, memory_key_padding_mask=visited_mask)  # (batch_size, 1, d_model)
            attn_output, attn_output_weights = self.mha(
                h, h_vertices, h_vertices, key_padding_mask=visited_mask, attn_mask=visited_mask.repeat_interleave(self.nhead, dim=0).unsqueeze(1)
            )  # (batch_size, 1, d_model), (batch_size, 1, graph_size), attn_mask is of shape (batch_size * nhead, 1, graph_size)
            attn_output_weights = attn_output_weights.squeeze(1)  # (batch_size, graph_size)
            if greedy:
                actions = torch.argmax(attn_output_weights, dim=1)  # (batch_size)
            else:
                actions = Categorical(attn_output_weights).sample()  # (batch_size)
            log_prob_list.append(torch.log(attn_output_weights[indexes, actions]))
            action_list.append(actions)
            h_last = h_vertices[indexes, actions]  # update part of state
            visited_mask[indexes, actions] = True  # update mask
        return torch.stack(action_list, dim=1), torch.stack(log_prob_list, dim=1).sum(dim=1)