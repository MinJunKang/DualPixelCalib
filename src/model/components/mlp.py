from typing import Optional, Set, Tuple, Union, List
import numpy as np

import torch
import torch.nn as nn
import tinycudann as tcnn


def activation_string_to_func(activation: str) -> nn.Module:
    if activation == 'ReLU':
        return nn.ReLU(inplace=True)
    elif activation == 'Leaky ReLU':
        return nn.LeakyReLU()
    elif activation == 'Sigmoid':
        return nn.Sigmoid()
    elif activation == 'Softplus':
        return nn.Softplus(beta=10)
    elif activation == 'Tanh':
        return nn.Tanh()
    elif activation == 'None':
        return None
    else:
        raise ValueError(f'Activation {activation} is not supported in MLP')



class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        scale_term: float = 0.001,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[str] = "None",
        weight_init: bool = False,
        weight_norm: bool = False,
        bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.num_layers = len(hidden_dims) + 1
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation_string_to_func(activation)
        self.weight_init = weight_init
        self.weight_norm = weight_norm
        self.bias = bias
        self.type = 'mlp'
        self.scale_term = scale_term
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            # first layer
            lin = nn.Linear(self.in_dim, self.out_dim)
            if self.weight_norm: lin = nn.utils.weight_norm(lin)
            layers.append(lin)
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    # first layer
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    input_dim, output_dim = self.in_dim, self.hidden_dims[i]
                elif i in self._skip_connections:
                    input_dim, output_dim = self.hidden_dims[i - 1] + self.in_dim, self.hidden_dims[i]
                else:
                    input_dim, output_dim = self.hidden_dims[i - 1], self.hidden_dims[i]
                    
                # layers
                lin = nn.Linear(input_dim, output_dim)
                if self.weight_init:
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(input_dim))
                    torch.nn.init.constant_(lin.bias, 0.0)
                if self.weight_norm: lin = nn.utils.weight_norm(lin)
                layers.append(lin)
                
            # final layer
            lin = nn.Linear(self.hidden_dims[-1], self.out_dim)
            if self.weight_init:
                torch.nn.init.normal_(lin.weight,
                                    mean=np.sqrt(np.pi) / np.sqrt(self.hidden_dims[-1]),
                                    std=0.0001)
                torch.nn.init.constant_(lin.bias, -self.bias)
            if self.weight_norm: lin = nn.utils.weight_norm(lin)
            layers.append(lin)
            
        self.layers = nn.ModuleList(layers)

    def forward(self, in_tensor):
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([x, in_tensor], dim=-1) / np.sqrt(2)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 2:
                x = self.activation(x)
        return x.abs() * self.scale_term
    
    
    
class TCNNMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        out_dim: int,
        scale_term: float = 0.001,
        activation: Optional[str] = "None",
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        
        self.net = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=out_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
            },
        )
        self.scale_term = scale_term
        
    def forward(self, in_tensor):
        return self.net(in_tensor).abs() * self.scale_term