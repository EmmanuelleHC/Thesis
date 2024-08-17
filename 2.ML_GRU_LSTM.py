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

    def prepare_graph_data(self):
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

def load_most_recent_csv(directory):
    list_of_files = glob.glob(f'{directory}/*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
def process_and_predict(model, filename, output_directory):
    # Load data
    df = pd.read_csv(filename)
    processor = GraphDataProcessor(df)
    test_data = processor.prepare_graph_data()
    test_loader = DataLoader([test_data], batch_size=1)

    # Predict
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            prediction = torch.sigmoid(output).cpu().numpy()  # Sigmoid to convert logits to probabilities
            predictions.extend(prediction)  # Using extend to flatten the array

    # Apply threshold to convert probabilities to binary labels
    threshold = 0.5
    predictions_label = [1 if p >= threshold else 0 for p in predictions]

    # Check if prediction length matches DataFrame rows
    if len(predictions) != len(df):
        raise ValueError(f"Expected {len(df)} predictions, but got {len(predictions)}")

    # Save predictions back to CSV
    df['Predictions'] = predictions  # Save probabilities
    df['Label_Prediction'] = predictions_label  # Save binary labels
    df.to_csv(filename, index=False)

    # Move the file to 'Result_Done' directory
    os.makedirs(output_directory, exist_ok=True)
    shutil.move(filename, os.path.join(output_directory, os.path.basename(filename)))


def process_all_csv(directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    for filename in glob.glob(f'{directory}/*.csv'):
        process_and_predict(model, filename, output_directory)
        print(f"File {os.path.basename(filename)} processed and moved to {output_directory}")

# Main execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('gcn_lstm_model.pth', map_location=device)
params = checkpoint['hyperparameters']
model = EdgeGCN_LSTM(hidden_channels=params['hidden_channels'], lstm_hidden_channels=params['lstm_hidden_channels'], out_channels=params['out_channels'], dropout_rate=params['dropout_rate'], num_layers=params['num_layers'], l2_lambda=params['l2_lambda'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

directory = 'Result'
output_directory = 'Result_Done'
process_all_csv(directory, output_directory)