import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
class EdgeGAT_GRU(nn.Module):
    def __init__(self, hidden_channels, gru_hidden_channels, out_channels, dropout_rate, num_layers, l2_lambda):
        super(EdgeGAT_GRU, self).__init__()
        self.conv1 = GATConv(1, hidden_channels)  # Use GATConv
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)  # Another GATConv
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.gru = nn.GRU(
            input_size=hidden_channels * 2 + 3,  # input dimensionality
            hidden_size=gru_hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        self.lin1 = nn.Linear(gru_hidden_channels, gru_hidden_channels // 2)
        self.lin2 = nn.Linear(gru_hidden_channels // 2, out_channels)
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))), p=self.dropout_rate, training=self.training)
        sender_features = x[edge_index[0]]
        receiver_features = x[edge_index[1]]
        edge_features = torch.cat([sender_features, receiver_features, edge_attr], dim=1)
        edge_features = edge_features.unsqueeze(0)
        gru_out, _ = self.gru(edge_features)
        gru_out = gru_out.squeeze(0)
        out = F.relu(self.lin1(gru_out))
        out = self.lin2(out)
        return out.view(-1)


class GraphDataProcessor:
    def __init__(self, df):
        self.df = df

    def undersample_df(self):
        fraud_df = self.df[self.df['Label'] == 1]
        non_fraud_df = self.df[self.df['Label'] == 0]
        balanced_df = non_fraud_df.sample(len(fraud_df), random_state=42)
        self.df = pd.concat([fraud_df, balanced_df])

    def prepare_graph_data(self):
        self.undersample_df()
        self.df['Time_step'] = pd.to_datetime(self.df['Time_step'])
        self.df = self.df.sort_values(by=['Sender_Customer_Id', 'Time_step'])

        self.df['Label'] = pd.to_numeric(self.df['Label'], errors='coerce').fillna(0).astype(int)

        all_ids = pd.concat([self.df['Sender_Customer_Id'], self.df['Bene_Customer_Id']]).unique()
        id_map = {id: idx for idx, id in enumerate(all_ids)}
        edge_index = torch.tensor(
            np.vstack([
                self.df['Sender_Customer_Id'].map(id_map).values,
                self.df['Bene_Customer_Id'].map(id_map).values
            ]), dtype=torch.long)
        node_features = torch.zeros((len(all_ids), 1))

        edge_attr = torch.cat([
            torch.tensor(LabelEncoder().fit_transform(self.df['Transaction_Type']), dtype=torch.float).view(-1, 1),
            torch.tensor(StandardScaler().fit_transform(self.df[['USD_Amount']]), dtype=torch.float).view(-1, 1),
            torch.tensor(self.df['risk_score'].values, dtype=torch.float).view(-1, 1)
        ], dim=1)
        edge_labels = torch.tensor(self.df['Label'].values, dtype=torch.long)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)
