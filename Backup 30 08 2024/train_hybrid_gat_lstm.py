import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_df = pd.read_csv('Thesis/train.csv')

# Setup Fuzzy Logic System
def setup_fuzzy_system():
    cross_border = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'cross_border')
    country_risk = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'country_risk')
    pep_involvement = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'pep_involvement')
    transaction_type = ctrl.Antecedent(np.arange(0, 3, 1), 'transaction_type')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    pep_involvement['no'] = fuzz.trapmf(pep_involvement.universe, [0, 0, 0.3, 0.5])
    pep_involvement['yes'] = fuzz.trapmf(pep_involvement.universe, [0.5, 0.7, 1, 1])
    cross_border['low'] = fuzz.trapmf(cross_border.universe, [0, 0, 0.3, 0.5])
    cross_border['high'] = fuzz.trapmf(cross_border.universe, [0.5, 0.7, 1, 1])
    country_risk['low'] = fuzz.trapmf(country_risk.universe, [0, 0, 0.3, 0.5])
    country_risk['high'] = fuzz.trapmf(country_risk.universe, [0.5, 0.7, 1, 1])
    transaction_type['crypto_transfer'] = fuzz.trimf(transaction_type.universe, [0, 0, 1])
    transaction_type['payment'] = fuzz.trimf(transaction_type.universe, [1, 1, 2])
    transaction_type['other'] = fuzz.trimf(transaction_type.universe, [2, 2, 2])
    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 50])
    risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 80])
    risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

    rule1 = ctrl.Rule(transaction_type['crypto_transfer'] | transaction_type['payment'], risk['high'])
    rule2 = ctrl.Rule(pep_involvement['yes'] | country_risk['high'], risk['high'])
    rule3 = ctrl.Rule(cross_border['high'], risk['medium'])
    rule4 = ctrl.Rule(cross_border['low'] & transaction_type['other'], risk['low'])

    aml_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    aml_sim = ctrl.ControlSystemSimulation(aml_control)
    return aml_sim

# Evaluate transaction risk using fuzzy logic
def evaluate_transaction(row, aml_sim):
    transaction_type_map = {'CRYPTO-TRANSFER': 0, 'PAYMENT': 1, 'OTHER': 2}
    transaction_type_value = transaction_type_map.get(row['Transaction_Type'], 2)
    pep_involvement_value = 1 if row['Bene_Is_Pep'] or row['Sender_Is_Pep'] else 0
    cross_border_value = 1 if row['Sender_Country'] != row['Bene_Country'] else 0
    high_risk_countries = ['Iran', 'Syria', 'North-Korea']
    country_risk_value = 1 if row['Bene_Country'] in high_risk_countries else 0

    aml_sim.input['transaction_type'] = transaction_type_value
    aml_sim.input['pep_involvement'] = pep_involvement_value
    aml_sim.input['cross_border'] = cross_border_value
    aml_sim.input['country_risk'] = country_risk_value

    aml_sim.compute()
    return float(aml_sim.output['risk'])

class EdgeGAT_LSTM(nn.Module):
    def __init__(self, hidden_channels, lstm_hidden_channels, out_channels, dropout_rate, lstm_layers):
        super(EdgeGAT_LSTM, self).__init__()
        self.gat1 = GATConv(1, hidden_channels, heads=4)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=4)
        self.gat3 = GATConv(hidden_channels * 4, hidden_channels, heads=4)
        
        self.lstm = nn.LSTM(
            input_size=hidden_channels * 4 + 3,  # Updated input_size
            hidden_size=lstm_hidden_channels,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.lin = nn.Linear(lstm_hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_attr):
        x = self.apply_gat_layers(x, edge_index)
        edge_features = self.prepare_edge_features(x, edge_index, edge_attr)
        lstm_out = self.apply_lstm(edge_features)
        out = self.lin(lstm_out)
        return out.view(-1)
    
    def apply_gat_layers(self, x, edge_index):
        x = F.dropout(F.relu(self.gat1(x, edge_index)), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.gat2(x, edge_index)), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.gat3(x, edge_index)), p=self.dropout_rate, training=self.training)
        return x
    
    def prepare_edge_features(self, x, edge_index, edge_attr):
        sender_features = x[edge_index[0]]
        receiver_features = x[edge_index[1]]
        edge_features = torch.cat([sender_features, receiver_features, edge_attr], dim=1)
        return edge_features.unsqueeze(0)  # Add batch dimension

    def apply_lstm(self, edge_features):
        lstm_out, _ = self.lstm(edge_features)
        return lstm_out.squeeze(0)  # Remove batch dimension
    
class GraphDataProcessor:
    def __init__(self, df):
        self.df = df
        self.aml_sim = setup_fuzzy_system()

    def undersample_df(self):
        fraud_df = self.df[self.df['Label'] == 1]
        non_fraud_df = self.df[self.df['Label'] == 0]
        print(f"Initial fraud cases: {len(fraud_df)}, non-fraud cases: {len(non_fraud_df)}")

        if len(fraud_df) < len(non_fraud_df):
            balanced_df = non_fraud_df.sample(len(fraud_df), random_state=42)
        else:
            balanced_df = non_fraud_df

        self.df = pd.concat([fraud_df, balanced_df]).sample(frac=1)  # shuffle the dataset
        print(f"Balanced dataset: {len(self.df)} records")

    def evaluate_risk_scores(self):
        self.df['Risk_Score'] = self.df.apply(lambda row: evaluate_transaction(row, self.aml_sim), axis=1)

    def prepare_graph_data(self):
        self.evaluate_risk_scores()
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
        risk_score = torch.tensor(self.df['Risk_Score'].values, dtype=torch.float).view(-1, 1)

        edge_attr = torch.cat([transaction_type_encoded, usd_amount, risk_score], dim=1)
        edge_labels = torch.tensor(self.df['Label'].values, dtype=torch.long)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.25,
    random_state=42,
    stratify=train_df['Label']
)

# Process data
train_processor = GraphDataProcessor(train_df)
val_processor = GraphDataProcessor(val_df)

train_data = train_processor.prepare_graph_data()
val_data = val_processor.prepare_graph_data()

train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
val_loader = DataLoader([val_data], batch_size=32, shuffle=False)

def train(model, device, loader, optimizer, criterion, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, data in enumerate(loader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(output, data.y.float()) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
    return total_loss / len(loader)

def evaluate(model, device, loader, criterion, threshold=0.5):
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
            preds = (probs > threshold).astype(int)

            y_scores.extend(probs)
            y_pred.extend(preds)
            y_true.extend(data.y.cpu().numpy())

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ks_statistic = max(tpr - fpr)

    return total_loss / len(loader), f1, precision, recall, auc, ks_statistic

accumulation_steps = 4
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64])
    lstm_hidden_channels = trial.suggest_categorical('lstm_hidden_channels', [16, 32, 64])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.7)
    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)

    model = EdgeGAT_LSTM(hidden_channels=hidden_channels, lstm_hidden_channels=lstm_hidden_channels, out_channels=1, dropout_rate=dropout_rate, lstm_layers=lstm_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    for epoch in range(10):
        train_loss = train(model, device, train_loader, optimizer, criterion, accumulation_steps)
        val_loss, f1, precision, recall, auc, ks_statistic = evaluate(model, device, val_loader, criterion)
        if f1 > best_f1:
            best_f1 = f1
            best_model_path = f"Thesis/best_model_trial_{trial.number}.pth"
            checkpoint = {
                'state_dict': model.state_dict(),
                'hyperparameters': {
                    'hidden_channels': hidden_channels,
                    'lstm_hidden_channels': lstm_hidden_channels,
                    'out_channels': 1,
                    'dropout_rate': dropout_rate,
                    'lstm_layers': lstm_layers
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

    return best_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Output best trial results
print("Best trial:")
trial = study.best_trial
print(f" Value (F1 Score): {trial.value}")
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Load best model and print metrics
best_trial_model_path = f"Thesis/best_model_trial_{study.best_trial.number}.pth"
checkpoint = torch.load(best_trial_model_path)
metrics = checkpoint['metrics']
print(" Validation set metrics:")
print(f"    F1 Score: {metrics['f1']}")
print(f"    Precision: {metrics['precision']}")
print(f"    Recall: {metrics['recall']}")
print(f"    AUC: {metrics['auc']}")
print(f"    KS Statistic: {metrics['ks_statistic']}")
