import torch
import torch_geometric
from torch_geometric.data import Batch

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch_geometric.nn as graphnn
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

def image_to_graph(
    images: torch.Tensor, conv2d: nn.Conv2d | None = None
) -> Batch:
    """
    Converts a batch of images to a PyTorch Geometric Batch object.

    Arguments:
    ----------
    images : torch.Tensor
        Image tensor of shape (B, C, H, W) or (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Convolution layer that determines kernel_size, stride, and padding (if desired).
        By default, None.

    Returns:
    --------
    torch_geometric.data.Batch
        A batched graph representation of all images.
    """
    # Check image dimension
    if images.dim() == 3:
        images = images.unsqueeze(0) # Batch_size = 1

    kernel_size = conv2d.kernel_size if conv2d is not None else (3, 3)
    stride = conv2d.stride if conv2d is not None else (1, 1)
    padding = conv2d.padding if conv2d is not None else (1, 1)

    B, C, H, W = images.shape
    data_list = []

    # image size after applying convolution
    H_o = (H + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_o = (W + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

    # padded image size
    H_padded, W_padded = H + 2 * padding[0], W + 2 * padding[1]

    # iterate over batch elements to bulid the graph representation for each image
    for i in range(B):
        image = images[i]

        padded_image = torch.nn.functional.pad(image, (padding[1], padding[1], padding[0], padding[0]))

        edge_index = []
        edge_attr = []

        for i in range(H_o):
            for j in range(W_o):
                start_i = i * stride[0]
                start_j = j * stride[1]

                # target node  
                target_idx = i * W_o + j

                # iterate over all possible source nodes
                for ki in range(kernel_size[0]):
                    for kj in range(kernel_size[1]):
                        src_i = start_i + ki
                        src_j = start_j + kj

                        if 0 <= src_i < H_padded and 0 <= src_j < W_padded:
                            src_idx = src_i * W_padded + src_j

                            edge_index.append([src_idx, target_idx])

                            kernel_pos = ki * kernel_size[1] + kj
                            edge_attr.append([kernel_pos])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        x = padded_image.permute(1, 2, 0).reshape(H_padded * W_padded, C)
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    return batch


def graph_to_image(
    data: torch.Tensor,
    height: int,
    width: int,
    conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of a batch of images to a 4D image tensor (B, C, H_out, W_out). B can be 1 for an individual image

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the batch of images. Shape should be (B*n_pixels, C) where n_pixels = H_o * W_o.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        (B, C, new_height, new_width),
        where B = 1 for a single graph, or batch.num_graphs for a Batch.
    """

    n_pixels, C = data.shape

    # extract kernel properties
    padding = conv2d.padding if conv2d is not None else (1, 1)
    kernel_size = conv2d.kernel_size if conv2d is not None else (3, 3)
    stride = conv2d.stride if conv2d is not None else (1, 1)
    
    # Padded dimensions
    H_padded = height + 2 * padding[0]
    W_padded = width + 2 * padding[1]

    # Image size after convolution
    H_o = (height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_o = (width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

    # Determine the batch_size: we consider two cases, that the image size match with the one after applying the convolution or it is the
    # one from the padded image, which can be the case if you do not apply the convolution and just want to retrieve the original image.
    # If the method instead of receiving the tensor corresponding to the data could receive the torch_geometric.data.Batch instance, we 
    # would directly ask for the batch size.
    padded_flag = False
    if n_pixels % (H_o * W_o) == 0:
      single_image_size = H_o * W_o
    elif n_pixels % (H_padded * W_padded) == 0:
      single_image_size = H_padded * W_padded
      padded_flag = True
    else:
      raise ValueError("n_pixels must be divisible by H_padded * W_padded or H_o * W_o")

    batch_size = n_pixels / single_image_size
    B = int(batch_size)

    images_list = []
    # iterate over the elements in the batch to retrieve the corresponding image
    for graph_idx in range(B):
        image = data[graph_idx*single_image_size:(graph_idx+1)*single_image_size]

        # Dealing with padded input features
        if padded_flag:
            padded_image = image.reshape(H_padded, W_padded, C)
            padded_image = padded_image.permute(2, 0, 1)

            image = padded_image[:, padding[0]:padding[0]+height, padding[1]:padding[1]+width]

        else: # Dealing with output features
            kernel_size = conv2d.kernel_size if conv2d is not None else (3, 3)
            stride = conv2d.stride if conv2d is not None else (1, 1)

            image = image.reshape(H_o, W_o, C)
            image = image.permute(2, 0, 1)

        images_list.append(image)

    # Stack into (B, C, new_height, new_width)
    out = torch.stack(images_list, dim=0)
    if B == 1:
        out = out.squeeze(0)

    return out

class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        super().__init__(aggr="add")

        # Store the Conv2d layer parameters
        self.conv2d = conv2d
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size
        self.padding = conv2d.padding
        self.stride = conv2d.stride

        # Extract weights and bias
        self.weight = conv2d.weight
        self.bias = conv2d.bias

    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size COMPLETE)
        """
        # Extract kernel positions from edge attributes
        kernel_pos = edge_attr[:, 0].long()

        # Count the number of incident edges for each target node
        max_index = torch.max(self.edge_index[1]) + 1
        num_incident_edges = torch.zeros(max_index, device=x_j.device)
        num_incident_edges.scatter_add_(0, self.edge_index[1], torch.ones_like(self.edge_index[1], dtype=torch.float))


        # Reshape and permute the weight tensor for efficient computation
        weights = self.weight.view(self.out_channels, self.in_channels, -1)
        weights = weights.permute(2, 0, 1)  # [kernel_size[0] * kernel_size[1], out_channels, in_channels]

        # Gather the appropriate weight slices
        weight_slices = weights[kernel_pos]  # [num_edges, out_channels, in_channels]

        # Apply weights to source feature
        messages = torch.einsum('eoi,ei->eo', weight_slices, x_j)  # [num_edges, out_channels]

        # Add bias divided by the number of incident edges if present (later in aggregation it will be added to each feature as the original bias value) 
        if self.bias is not None:
            target_indices = self.edge_index[1]
            bias_divided = self.bias / num_incident_edges[target_indices].unsqueeze(1)
            messages += bias_divided

        return messages

    # with this method we make sure that it will retrieve the representation of the image after applying the convolution, removing all
    # the rest of the pixels
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Get unique target nodes (output nodes)
        target_nodes = torch.unique(self.edge_index[1])

        # Extract only the aggregated output for target nodes
        output_features = aggr_out[target_nodes]

        return output_features