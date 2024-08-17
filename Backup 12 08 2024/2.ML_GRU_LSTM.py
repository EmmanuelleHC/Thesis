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
    def __init__(self, hidden_channels, lstm_hidden_channels, out_channels, dropout_rate, num_layers, l2_lambda):
        super(EdgeGCN_LSTM, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lstm = nn.LSTM(
            input_size=hidden_channels * 2 + 3,
            hidden_size=lstm_hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        self.lin1 = nn.Linear(lstm_hidden_channels, lstm_hidden_channels // 2)
        self.lin2 = nn.Linear(lstm_hidden_channels // 2, out_channels)
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))), p=self.dropout_rate, training=self.training)
        sender_features = x[edge_index[0]]
        receiver_features = x[edge_index[1]]
        edge_features = torch.cat([sender_features, receiver_features, edge_attr], dim=1)
        edge_features = edge_features.unsqueeze(0)
        lstm_out, _ = self.lstm(edge_features)
        lstm_out = lstm_out.squeeze(0)
        out = F.relu(self.lin1(lstm_out))
        out = self.lin2(out)
        return out.view(-1)

class GraphDataProcessor:
    def __init__(self, df):
        self.df = df

    def process_data(self):
        # Preprocess and prepare graph data here
        self.df['Transaction_Type'] = LabelEncoder().fit_transform(self.df['Transaction_Type'])
        self.df['USD_Amount'] = StandardScaler().fit_transform(self.df[['USD_Amount']])
        self.df['risk_score'] = StandardScaler().fit_transform(self.df[['risk_score']])
        
        # Prepare graph structure
        all_ids = pd.concat([self.df['Sender_Customer_Id'], self.df['Bene_Customer_Id']]).unique()
        id_map = {id: idx for idx, id in enumerate(all_ids)}
        edge_index = torch.tensor(
            np.vstack([
                self.df['Sender_Customer_Id'].map(id_map).values,
                self.df['Bene_Customer_Id'].map(id_map).values
            ]), dtype=torch.long)
        node_features = torch.zeros((len(all_ids), 1))
        edge_attr = torch.tensor(self.df[['Transaction_Type', 'USD_Amount', 'risk_score']].values, dtype=torch.float)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def load_most_recent_csv(directory):
    list_of_files = glob.glob(f'{directory}/*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def process_and_predict(model, filename, output_directory):
    # Load data
    df = pd.read_csv(filename)
    processor = GraphDataProcessor(df)
    test_data = processor.process_data()
    test_loader = DataLoader([test_data], batch_size=1)

    # Predict
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            prediction = torch.sigmoid(output).cpu().numpy()
            predictions.append(prediction[0])

    # Save predictions back to CSV
    df['Predictions'] = predictions
    df.to_csv(filename, index=False)

    # Move the file to 'Result_Done' directory
    os.makedirs(output_directory, exist_ok=True)
    shutil.move(filename, os.path.join(output_directory, os.path.basename(filename)))

# Main execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('gcn_lstm_model.pth', map_location=device)
params = checkpoint['params']

model = EdgeGCN_LSTM(
    hidden_channels=params['hidden_channels'],
    lstm_hidden_channels=params['lstm_hidden_channels'],
    out_channels=params['out_channels'],
    dropout_rate=params['dropout_rate'],
    num_layers=params['num_layers'],
    l2_lambda=params['l2_lambda']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
directory = 'Result'
output_directory = 'Result_Done'
latest_file = load_most_recent_csv(directory)
process_and_predict(model, latest_file, output_directory)
print(f"File {os.path.basename(latest_file)} processed and moved to {output_directory}")
