import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, att_heads, activation = nn.LeakyReLU(), dropout_prob=1, concat=True, add_skip_connection=True):
        """
        Multi-head attention mechanism for GAT.
        Args:
            input_dim: Input feature size per node
            hidden_dim: Feature size per head
            att_heads: Number of attention heads
            activation: Activation function
            dropout_prob: Dropout probability
            concat: Whether to concatenate or average attention heads
            add_skip_connection: Whether to add skip connection or not
        """
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.att_heads = att_heads
        self.activation = activation
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        # Multi-head projection layer (each head gets its own transformation)
        self.projection = nn.Linear(input_dim, att_heads * hidden_dim, bias=False)

        # Attention mechanism per head (2 * hidden_dim per head)
        self.attention_e_weights = nn.Parameter(torch.Tensor(1, self.att_heads, 2 * hidden_dim))  # Attention weights: a in GAT paper

        if add_skip_connection:
            self.skip_proj = nn.Linear(input_dim, hidden_dim * att_heads, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.dropout = nn.Dropout(p=dropout_prob) # In the original GAT paper, for PPI it is not applied, we won't use it

        self.initialize_parameters()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def initialize_parameters(self):
        """Apply Glorot (Xavier) initialization to weights.""" # Following the original GAT implementation
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_e_weights)
        if self.add_skip_connection:
            nn.init.xavier_uniform_(self.skip_proj.weight)

    def forward(self, x, edge_index):
        """
        Compute multi-head attention.
        Args:
            x: Node feature matrix of shape (batch_size, num_nodes, input_dim).
            edge_index: Edge indices (batch_size, 2, num_edges).
        Returns:
            h_prime: Updated node features (batch_size, num_nodes, att_heads * hidden_dim).
        """
        num_nodes, _ = x.shape

        # Apply linear projection to obtain multi-head representations
        h = self.projection(x)  # (batch_size, num_nodes, att_heads * hidden_dim)
        h = h.view(num_nodes, self.att_heads, self.hidden_dim)  # (batch_size, num_nodes, att_heads, hidden_dim)

        # Extract source & destination node indices
        src, dst = edge_index  # (2, num_edges)

        # Gather features for source and destination nodes
        h_src = h[src]  # (batch_size, num_edges, att_heads, hidden_dim)
        h_dst = h[dst]  # (batch_size, num_edges, att_heads, hidden_dim)

        # Concatenate source and destination embeddings along the feature dimension
        concat_h = torch.cat([h_src, h_dst], dim=-1)  # (batch_size, num_edges, att_heads, 2 * hidden_dim)

        # Compute raw attention scores
        e = torch.einsum("nah,nah -> na", concat_h, self.attention_e_weights)  # (batch_size, num_edges, att_heads)

        # Apply LeakyReLU activation to attention scores
        e = self.activation(e)

        # Compute softmax per destination node
        alpha = softmax(e, dst, num_nodes=num_nodes)  # (batch_size, num_edges, att_heads)

        # Compute weighted sum of source node features (using attention scores)
        h_prime = torch.zeros((num_nodes, self.att_heads, self.hidden_dim), device=x.device)
        h_prime.index_add_(0, dst, alpha.unsqueeze(-1) * h_src)

        # Apply skip connection (if enabled)
        if self.add_skip_connection:
            skip_output = self.skip_proj(x).view(num_nodes, self.att_heads, self.hidden_dim)
            h_prime += skip_output

        # Concatenate or average attention heads
        if self.concat:
            h_prime = h_prime.view(num_nodes, self.att_heads * self.hidden_dim)  # (batch_size, num_nodes, att_heads * hidden_dim)
        else:
            h_prime = h_prime.mean(dim=1)  # Average over heads: (num_nodes, hidden_dim)

        return h_prime  # Updated node features



# Define model ( in your class_model_gnn.py)
class StudentModel(nn.Module):
    def __init__(self, input_dim=50, hidden_dim = [256, 256, 121], att_heads = [4, 4, 6]):
        super(StudentModel, self).__init__()
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.att1 = MultiHeadAttention(input_dim, hidden_dim[0], att_heads[0], concat=True, add_skip_connection=True)
        self.att2 = MultiHeadAttention(hidden_dim[0] * att_heads[0], hidden_dim[1], att_heads[1], concat=True, add_skip_connection=True)
        self.att3 = MultiHeadAttention(hidden_dim[1] * att_heads[1], hidden_dim[2], att_heads[2], concat=False, add_skip_connection=True)

    def forward(self, x, edge_index):
        x = nn.ELU()(self.att1(x, edge_index))
        x = nn.ELU()(self.att2(x, edge_index))
        x = self.att3(x, edge_index)#.mean(dim=-1) # Paper indicates to do sigmoid, it is done in the loss function
                                                   # Also, mean(dim=-1) removed due to incompatible shape with target in BCE
        return x


# Initialize model
model = StudentModel()

## Save the model
torch.save(model.state_dict(), "model.pth")


### This is the part we will run in the inference to grade your model
## Load the model
model = StudentModel()  # !  Important : No argument
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
print("Model loaded successfully")
