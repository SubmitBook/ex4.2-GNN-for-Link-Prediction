import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE, GATConv, SAGEConv
from torch_geometric.utils import negative_sampling, train_test_split_edges

# Example graph with 6 nodes and undirected edges
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)

# Node features (randomly initialized for now)
x = torch.randn((6, 16))  # 6 nodes, each with a 16-dimensional feature vector

data = Data(x=x, edge_index=edge_index)

# Split edges for training/testing (needed for link prediction)
data = train_test_split_edges(data)

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type="GraphSAGE"):
        super().__init__()
        if model_type == "GraphSAGE":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif model_type == "GAT":
            self.conv1 = GATConv(in_channels, hidden_channels, heads=2)
            self.conv2 = GATConv(hidden_channels * 2, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels * 2, 1)

    def forward(self, x, edge_index):
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        return torch.sigmoid(self.fc(torch.cat([x_i, x_j], dim=1))).squeeze()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNNModel(in_channels=16, hidden_channels=32, out_channels=16, model_type="GraphSAGE").to(device)
predictor = LinkPredictor(16).to(device)

optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)
criterion = nn.BCELoss()

data = data.to(device)

for epoch in range(100):
    model.train()
    predictor.train()

    optimizer.zero_grad()
    x = model(data.x, data.train_pos_edge_index)

    # Positive and negative edges
    pos_pred = predictor(x, data.train_pos_edge_index)
    neg_edge_index = negative_sampling(data.train_pos_edge_index, num_nodes=data.num_nodes,
                                       num_neg_samples=pos_pred.size(0))
    neg_pred = predictor(x, neg_edge_index)

    loss = criterion(pos_pred, torch.ones_like(pos_pred)) + criterion(neg_pred, torch.zeros_like(neg_pred))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")