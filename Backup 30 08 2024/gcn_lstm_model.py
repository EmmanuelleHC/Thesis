import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import shutil
class EdgeGCN_LSTM(nn.Module):
    def __init__(self, hidden_channels, lstm_hidden_channels, out_channels, dropout_rate):
        super(EdgeGCN_LSTM, self).__init__()
        # Use a single GCN layer followed by LSTM
        self.conv1 = GCNConv(1, hidden_channels)
        self.lstm = nn.LSTM(input_size=hidden_channels * 2 + 3, hidden_size=lstm_hidden_channels, batch_first=True)
        self.lin = nn.Linear(lstm_hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_attr):
        # Apply GCN and dropout
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout_rate, training=self.training)
        
        # Prepare features for LSTM
        sender_features = x[edge_index[0]]
        receiver_features = x[edge_index[1]]
        edge_features = torch.cat([sender_features, receiver_features, edge_attr], dim=1)
        
        # Process with LSTM
        edge_features = edge_features.unsqueeze(0)  # Add batch dimension for LSTM
        lstm_out, _ = self.lstm(edge_features)
        lstm_out = lstm_out.squeeze(0)  # Remove batch dimension
        
        # Linear output layer
        out = self.lin(lstm_out)
        return out.view(-1)


class GraphDataProcessor:
    def __init__(self, df):
        self.df = df
    def undersample_df(self):
        fraud_df = self.df[self.df['Label'] == 1]
        non_fraud_df = self.df[self.df['Label'] == 0]
        print(f"Initial fraud cases: {len(fraud_df)}, non-fraud cases: {len(non_fraud_df)}")

          # Check if there are enough fraud cases to sample
        if len(fraud_df) < len(non_fraud_df):
              balanced_df = non_fraud_df.sample(len(fraud_df), random_state=42)
        else:
              balanced_df = non_fraud_df

        self.df = pd.concat([fraud_df, balanced_df]).sample(frac=1)  # shuffle the dataset
        print(f"Balanced dataset: {len(self.df)} records")

    def prepare_graph_data(self):
        self.undersample_df()
        self.df['Time_step'] = pd.to_datetime(self.df['Time_step'])
        self.df = self.df.sort_values(by=['Sender_Customer_Id', 'Time_step'])
        self.df['Label'] = pd.to_numeric(self.df['Label'], errors='coerce').fillna(0).astype(int)

        all_ids = pd.concat([self.df['Sender_Customer_Id'], self.df['Bene_Customer_Id']]).unique()
        if len(all_ids) == 0:
            raise ValueError("No unique IDs found in the dataset"+len(self.df))

        id_map = {id: idx for idx, id in enumerate(all_ids)}
        edge_index = torch.tensor([self.df['Sender_Customer_Id'].map(id_map).values, self.df['Bene_Customer_Id'].map(id_map).values], dtype=torch.long)

        node_features = torch.zeros((len(all_ids), 1))
      
        transaction_type_encoded = torch.tensor(LabelEncoder().fit_transform(self.df['Transaction_Type']), dtype=torch.float).view(-1, 1)
        usd_amount = torch.tensor(StandardScaler().fit_transform(self.df[['USD_Amount']]), dtype=torch.float).view(-1, 1)
        risk_score = torch.tensor(self.df['risk_score'].values, dtype=torch.float).view(-1, 1)

        edge_attr = torch.cat([transaction_type_encoded, usd_amount, risk_score], dim=1)
        edge_labels = torch.tensor(self.df['Label'].values, dtype=torch.long)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)
