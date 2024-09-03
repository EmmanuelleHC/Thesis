import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score,roc_curve
import optuna
import numpy as np
from sklearn.model_selection import train_test_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
train_df = pd.read_csv('Thesis/train_with_fuzzy_results2.csv')

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
            raise ValueError("No unique IDs found in the dataset")

        id_map = {id: idx for idx, id in enumerate(all_ids)}
        edge_index = torch.tensor([self.df['Sender_Customer_Id'].map(id_map).values, self.df['Bene_Customer_Id'].map(id_map).values], dtype=torch.long)

        node_features = torch.zeros((len(all_ids), 1))
      
        transaction_type_encoded = torch.tensor(LabelEncoder().fit_transform(self.df['Transaction_Type']), dtype=torch.float).view(-1, 1)
        usd_amount = torch.tensor(StandardScaler().fit_transform(self.df[['USD_Amount']]), dtype=torch.float).view(-1, 1)
        risk_score = torch.tensor(self.df['risk_score'].values, dtype=torch.float).view(-1, 1)

        edge_attr = torch.cat([transaction_type_encoded, usd_amount, risk_score], dim=1)
        edge_labels = torch.tensor(self.df['Label'].values, dtype=torch.long)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)


train_df, val_df = train_test_split(
    train_df,
    test_size=0.25,
    random_state=42,
    stratify=train_df['Label']
)

train_processor = GraphDataProcessor(train_df)
val_processor = GraphDataProcessor(val_df)

train_data = train_processor.prepare_graph_data()
val_data = val_processor.prepare_graph_data()

train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
val_loader = DataLoader([val_data], batch_size=32, shuffle=False)

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(output, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, device, loader, criterion):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(output, data.y.float())
            total_loss += loss.item()

            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > 0.4).astype(int)

            y_scores.extend(probs)
            y_pred.extend(preds)
            y_true.extend(data.y.cpu().numpy())

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    ks_statistic = max(tpr - fpr)

    return total_loss / len(loader), f1, precision, recall, auc, ks_statistic
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64])
    lstm_hidden_channels = trial.suggest_categorical('lstm_hidden_channels', [16, 32, 64])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.7)

    model = EdgeGCN_LSTM(hidden_channels=hidden_channels, lstm_hidden_channels=lstm_hidden_channels, out_channels=1, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    for epoch in range(10):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss, f1, precision, recall, auc, ks_statistic = evaluate(model, device, val_loader, criterion)
        if f1 > best_f1:
            best_f1 = f1
            best_model_path = f"Thesis/best_model_trial_{trial.number}.pth"
            # Save both model state and hyperparameters
            checkpoint = {
                'state_dict': model.state_dict(),
                'hyperparameters': {
                    'hidden_channels': hidden_channels,
                    'lstm_hidden_channels': lstm_hidden_channels,
                    'out_channels': 1,
                    'dropout_rate': dropout_rate
                },
                'metrics': {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc,
                    'ks_statistic': ks_statistic
                }
            }
            torch.save(checkpoint, best_model_path)
            # Assume manual upload to Google Drive or use a sync mechanism

    return best_f1  # Optimize for the best F1 score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial
print(f" Value (F1 Score): {trial.value}")
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Assuming the path is a local path where Google Drive syncs or is manually uploaded
best_trial_model_path = f"Thesis/best_model_trial_{study.best_trial.number}.pth"

# Load the best model from a local path, assuming it is synced to Google Drive
checkpoint = torch.load(best_trial_model_path)
metrics = checkpoint['metrics']
print(" Validation set metrics:")
print(f"    F1 Score: {metrics['f1']}")
print(f"    Precision: {metrics['precision']}")
print(f"    Recall: {metrics['recall']}")
print(f"    AUC: {metrics['auc']}")
print(f"    KS Statistic: {metrics['ks_statistic']}")
