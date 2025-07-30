import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv, SAGEConv, MessagePassing

class GNN_Model(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out=1,
        num_layers=2,
        dp_rate=0.1,
        SAGE=False
        ):
        """GNNModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network

        """
        super().__init__()
        

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                SAGEConv(in_channels=in_channels, out_channels=out_channels) if SAGE else GCNConv(in_channels=in_channels, out_channels=out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [GCNConv(in_channels=in_channels, out_channels=c_out)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)

        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
    