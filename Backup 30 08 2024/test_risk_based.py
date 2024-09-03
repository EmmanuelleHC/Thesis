import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import ks_2samp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Add these imports

class EdgeGCN_LSTM(nn.Module):
    def __init__(self, hidden_channels, lstm_hidden_channels, out_channels, dropout_rate, num_layers, l2_lambda):
        super(EdgeGCN_LSTM, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lstm = nn.LSTM(
            input_size=hidden_channels * 2 + 2,
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

    def undersample_df(self):
        fraud_df = self.df[self.df['Label'] == 1]
        non_fraud_df = self.df[self.df['Label'] == 0]
        balanced_df = non_fraud_df.sample(len(fraud_df), random_state=42)
        self.df = pd.concat([fraud_df, balanced_df])

    def prepare_graph_data(self):
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
        ], dim=1)
        edge_labels = torch.tensor(self.df['Label'].values, dtype=torch.long)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = EdgeGCN_LSTM(
        hidden_channels=checkpoint['hyperparameters']['hidden_channels'],
        lstm_hidden_channels=checkpoint['hyperparameters']['lstm_hidden_channels'],
        out_channels=checkpoint['hyperparameters']['out_channels'],
        dropout_rate=checkpoint['hyperparameters']['dropout_rate'],
        num_layers=checkpoint['hyperparameters']['num_layers'],
        l2_lambda=checkpoint['hyperparameters']['l2_lambda']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def test(model, device, loader, criterion):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    total_loss = 0
    all_probs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(output, data.y.float())
            total_loss += loss.item()

            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            y_scores.extend(probs)  # Collect probabilities for AUC and KS calculation
            y_pred.extend(preds)
            y_true.extend(data.y.cpu().numpy())

            # Collect probabilities in a structured way for KS
            all_probs.append(probs)

    # Concatenate all probabilities collected
    all_probs = np.concatenate(all_probs)
    fraud_probs = all_probs[np.array(y_true) == 1]
    non_fraud_probs = all_probs[np.array(y_true) == 0]
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    ks_stat, ks_pvalue = ks_2samp(non_fraud_probs, fraud_probs)

    return total_loss / len(loader), f1, precision, recall, auc, ks_stat, ks_pvalue


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_df = pd.read_csv('Thesis/test.csv')  # Modify path as needed
# Load the model
model = load_model('risk_based.pth')
model = model.to(device)

# Process the test data
test_processor = GraphDataProcessor(test_df)
test_data = test_processor.prepare_graph_data()
test_loader = DataLoader([test_data], batch_size=32, shuffle=False)

# Criterion for testing
criterion = torch.nn.BCEWithLogitsLoss()

# Evaluate the model
test_loss, test_f1, test_precision, test_recall, test_auc, test_ks_stat, test_ks_pvalue = test(model, device, test_loader, criterion)
print(f"Test Loss: {test_loss}")
print(f"F1 Score: {test_f1}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"AUC: {test_auc}")
print(f"KS Statistic: {test_ks_stat}")
print(f"KS P-value: {test_ks_pvalue}")
