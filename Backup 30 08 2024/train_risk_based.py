import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import optuna
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
train_df = pd.read_csv('Thesis/train.csv')

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
        ], dim=1)
        edge_labels = torch.tensor(self.df['Label'].values, dtype=torch.long)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

def split_data(df, val_size=0.2):
    fraud_df = df[df['Label'] == 1]
    non_fraud_df = df[df['Label'] == 0]
    
    # Split non-fraudulent transactions into train and validation sets
    non_fraud_train_df = non_fraud_df.sample(frac=1 - val_size, random_state=42)
    non_fraud_val_df = non_fraud_df.drop(non_fraud_train_df.index)
    
    # Split fraudulent transactions into train and validation sets
    fraud_train_df = fraud_df.sample(frac=1 - val_size, random_state=42)
    fraud_val_df = fraud_df.drop(fraud_train_df.index)
    
    train_df = pd.concat([fraud_train_df, non_fraud_train_df])
    val_df = pd.concat([fraud_val_df, non_fraud_val_df])
    
    return train_df, val_df

def train(model, device, loader, optimizer, criterion, l2_lambda):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(output, data.y.float())
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param)
        loss = loss + l2_lambda * l2_reg
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
            preds = (probs > 0.5).astype(int)
            y_scores.extend(probs)
            y_pred.extend(preds)
            y_true.extend(data.y.cpu().numpy())
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    return total_loss / len(loader), f1, precision, recall, auc

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64])
    lstm_hidden_channels = trial.suggest_categorical('lstm_hidden_channels', [16, 32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.7)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    l2_lambda = trial.suggest_loguniform('l2_lambda', 1e-6, 1e-2)

    model = EdgeGCN_LSTM(hidden_channels=hidden_channels, lstm_hidden_channels=lstm_hidden_channels,
                         out_channels=1, dropout_rate=dropout_rate, num_layers=num_layers, l2_lambda=l2_lambda).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Split data into training and validation sets
    train_data_df, val_data_df = split_data(train_df)

    # Prepare train and validation data
    train_processor = GraphDataProcessor(train_data_df)
    val_processor = GraphDataProcessor(val_data_df)
    train_data = train_processor.prepare_graph_data()
    val_data = val_processor.prepare_graph_data()
    
    train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=32, shuffle=False)

    best_val_f1 = 0
    for epoch in range(10):  # Adjust the number of epochs if needed
        train_loss = train(model, device, train_loader, optimizer, criterion, l2_lambda)
        val_loss, val_f1, val_precision, val_recall, val_auc = evaluate(model, device, val_loader, criterion)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    return best_val_f1

# Optuna study to find best hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print("Best trial:")
trial = study.best_trial
print(f" Value (F1 Score): {trial.value}")
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Retrain with the best hyperparameters and save the model
best_params = trial.params
model = EdgeGCN_LSTM(
    hidden_channels=best_params['hidden_channels'],
    lstm_hidden_channels=best_params['lstm_hidden_channels'],
    out_channels=1,
    dropout_rate=best_params['dropout_rate'],
    num_layers=best_params['num_layers'],
    l2_lambda=best_params['l2_lambda']
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.BCEWithLogitsLoss()

# Prepare train and validation data again with the best params
train_data_df, val_data_df = split_data(train_df)
train_processor = GraphDataProcessor(train_data_df)
val_processor = GraphDataProcessor(val_data_df)
train_data = train_processor.prepare_graph_data()
val_data = val_processor.prepare_graph_data()

train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
val_loader = DataLoader([val_data], batch_size=32, shuffle=False)

# Train the final model and print metrics
for epoch in range(10):  # Adjust the number of epochs if needed
    train_loss = train(model, device, train_loader, optimizer, criterion, best_params['l2_lambda'])
    val_loss, val_f1, val_precision, val_recall, val_auc = evaluate(model, device, val_loader, criterion)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val AUC: {val_auc:.4f}")

# Final evaluation on the validation set
final_val_loss, final_val_f1, final_val_precision, final_val_recall, final_val_auc = evaluate(model, device, val_loader, criterion)

# Print the final metrics
print("\nFinal Evaluation Metrics on Validation Set:")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Final Validation F1 Score: {final_val_f1:.4f}")
print(f"Final Validation Precision: {final_val_precision:.4f}")
print(f"Final Validation Recall: {final_val_recall:.4f}")
print(f"Final Validation AUC: {final_val_auc:.4f}")

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': {
        'hidden_channels': best_params['hidden_channels'],
        'lstm_hidden_channels': best_params['lstm_hidden_channels'],
        'out_channels': 1,
        'dropout_rate': best_params['dropout_rate'],
        'num_layers': best_params['num_layers'],
        'l2_lambda': best_params['l2_lambda'],
        'lr': best_params['lr']
    }
}, 'risk_based.pth')
print("Model saved as 'risk_based.pth'")
