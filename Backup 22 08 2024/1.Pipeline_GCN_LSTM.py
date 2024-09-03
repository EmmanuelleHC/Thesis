import json
import os
import threading
import queue
from datetime import datetime
import time
import logging
import shutil
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from kafka import KafkaConsumer
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_fuzzy_system():
    # Define fuzzy logic system
    cross_border = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'cross_border')
    country_risk = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'country_risk')
    pep_involvement = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'pep_involvement')
    transaction_type = ctrl.Antecedent(np.arange(0, 3, 1), 'transaction_type')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # Membership Functions
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

    # Rules
    rule1 = ctrl.Rule(transaction_type['crypto_transfer'] | transaction_type['payment'], risk['high'])
    rule2 = ctrl.Rule(pep_involvement['yes'] | country_risk['high'], risk['high'])
    rule3 = ctrl.Rule(cross_border['high'], risk['medium'])
    rule4 = ctrl.Rule(cross_border['low'] & transaction_type['other'], risk['low'])

    # Control system
    aml_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    aml_sim = ctrl.ControlSystemSimulation(aml_control)
    return aml_sim



def evaluate_transaction(row, aml_sim):
    transaction_type_map = {'CRYPTO-TRANSFER': 1, 'PAYMENT': 1, 'OTHER': 2}
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
    risk_score = aml_sim.output['risk']

    # Determine reasons for risk score
    reasons = []
    if transaction_type_value == 1:
        reasons.append('Transaction type: High-risk transaction (crypto transfer/payment)')
    if pep_involvement_value == 1:
        reasons.append('PEP involvement')
    if cross_border_value == 1:
        reasons.append('Cross-border transaction')
    if country_risk_value == 1:
        reasons.append('High-risk country involvement')

    return risk_score, reasons

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
        self.aml_sim = setup_fuzzy_system()

    def evaluate_risk_scores(self):
        results = self.df.apply(lambda row: evaluate_transaction(row, self.aml_sim), axis=1)
        self.df['Risk_Score'], self.df['Risk_Reasons'] = zip(*results)

    def prepare_graph_data(self):
        self.evaluate_risk_scores()

        self.df['Time_step'] = pd.to_datetime(self.df['Time_step'])
        self.df = self.df.sort_values(by=['Sender_Customer_Id', 'Time_step'])
        self.df['Label'] = pd.to_numeric(self.df['Label'], errors='coerce').fillna(0).astype(int)

    
        all_ids = pd.concat([self.df['Sender_Customer_Id'], self.df['Bene_Customer_Id']]).unique()
        if len(all_ids) == 0:
            raise ValueError("No unique IDs found in the dataset")

        id_map = {id: idx for idx, id in enumerate(all_ids)}

        # Check if mapping produces valid outputs
        sender_customer_mapped = self.df['Sender_Customer_Id'].map(id_map).values
        bene_customer_mapped = self.df['Bene_Customer_Id'].map(id_map).values
        logging.info(f"Sender Customer Mapped Values: {sender_customer_mapped}")
        logging.info(f"Bene Customer Mapped Values: {bene_customer_mapped}")
        if pd.isnull(sender_customer_mapped).any() or pd.isnull(bene_customer_mapped).any():
            raise ValueError("Mapping resulted in NaN values")

        edge_index = torch.tensor(
            np.array([
                sender_customer_mapped,
                bene_customer_mapped
            ]), 
            dtype=torch.long
        )
        logging.info(f"edge_index shape: {edge_index.shape}, values: {edge_index}")

        node_features = torch.zeros((len(all_ids), 1))
        logging.info(f"node_features shape: {node_features.shape}")

        transaction_type_encoded = torch.tensor(
            pd.to_numeric(LabelEncoder().fit_transform(self.df['Transaction_Type']), errors='coerce').astype(float),
            dtype=torch.float
        ).view(-1, 1)
        logging.info(f"transaction_type_encoded shape: {transaction_type_encoded.shape}")
        self.df['USD_Amount'] = pd.to_numeric(self.df['USD_Amount'], errors='coerce')
        if self.df['USD_Amount'].isnull().any():
            logging.warning("NaN values found in 'USD_Amount' column")
            self.df['USD_Amount'].fillna(0, inplace=True)

        usd_amount_np = StandardScaler().fit_transform(self.df[['USD_Amount']].astype(float))
        logging.debug(f"USD Amount NumPy Array Shape: {usd_amount_np.shape}")

        usd_amount = torch.tensor(usd_amount_np, dtype=torch.float)
        usd_amount = usd_amount.view(-1, 1)
        

        risk_score = torch.tensor(
            pd.to_numeric(self.df['Risk_Score'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x),
            errors='coerce').astype(float),
            dtype=torch.float
        ).view(-1, 1)
        logging.debug(f"risk_score shape: {risk_score.shape}, values: {risk_score[:5]}")

        edge_attr = torch.cat([transaction_type_encoded, usd_amount, risk_score], dim=1)
        logging.debug(f"edge_attr shape: {edge_attr.shape}")

        edge_labels = torch.tensor(
            pd.to_numeric(self.df['Label'], errors='coerce').astype(int).values,
            dtype=torch.long
        )
        logging.debug(f"edge_labels shape: {edge_labels.shape}, values: {edge_labels[:5]}")

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

def create_consumer():
    return KafkaConsumer(
        'kraft-test',
        bootstrap_servers=['m3-login3.massive.org.au:9092'],
        auto_offset_reset='latest',  # Start consuming new messages only
        enable_auto_commit=True,  # Enable auto committing offsets
        auto_commit_interval_ms=1000,  # Commit every second
        fetch_max_wait_ms=4000  # Max wait time for data in fetch requests
    )
def consumer_thread(msg_queue, timeout_minutes=5):
    consumer = create_consumer()
    start_time = time.time()
    batch_count = 0  # Initialize a counter for batches received

    while time.time() - start_time < timeout_minutes * 60:
        consumer.poll(timeout_ms=500)
        for message in consumer:
            msg_queue.put(message.value)  # Only put the value of the message into the queue
            batch_count += 1  # Increment the batch counter for each message received

    msg_queue.put(None)  # Signal that consumption is done
    logging.info(f"Total batches received: {batch_count}")

def processing_thread(msg_queue, model, device, output_directory):
    while True:
        message = msg_queue.get()  # Retrieve message from the queue
        if message is None:  # Check if the consumer has finished
            break  

        # Process the message using the provided model and device
        labels, predictions, processed_df = process_message(message, model, device, output_directory)

        if processed_df is not None:
            threshold = 0.5

            # Convert continuous predictions to binary labels using the threshold
            predictions_label = (predictions >= threshold).astype(int)

            # Calculate metrics
            precision = precision_score(labels, predictions_label, zero_division=0)
            recall = recall_score(labels, predictions_label, zero_division=0)
            f1 = f1_score(labels, predictions_label, zero_division=0)

            # Conditional AUC calculation
            auc, ks_statistic = None, None
            if len(np.unique(labels)) > 1:
                try:
                    auc = roc_auc_score(labels, predictions)
                    fpr, tpr, thresholds = roc_curve(labels, predictions)
                    ks_statistic = max(tpr - fpr)
                except ValueError as e:
                    logging.warning(f"Failed to compute AUC or KS statistic: {e}")
            else:
                logging.warning("Cannot compute ROC AUC and KS statistic because only one class is present.")

            # Log and save the metrics
            log_and_save_metrics(output_directory, precision, recall, f1, auc, ks_statistic)

        # Indicate that processing is complete for this message
        logging.info("Completed processing a message")

    logging.info("All messages processed")



def log_and_save_metrics(output_directory, precision, recall, f1, auc, ks_stat):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Metrics_{timestamp}.csv"
    filepath = os.path.join(output_directory, filename)
    metrics_data = pd.DataFrame({
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'AUC': [auc],
        'KS_Statistic': [ks_stat],
    })
    metrics_data.to_csv(filepath, index=False)
    logging.info(f"Metrics saved to {filepath}")
def process_message(message, model, device, output_directory):
    try:
        logging.info("Decoding JSON message")
        decoded_message = json.loads(message.decode('utf-8'))
        df = pd.DataFrame(decoded_message)
        logging.info(f"DataFrame created with {len(df)} rows")

        logging.info("Preparing graph data")
        processor = GraphDataProcessor(df)
        graph_data = processor.prepare_graph_data().to(device)
        logging.info(f"Graph data prepared with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")

        logging.info("Evaluating model")
        model.eval()
        with torch.no_grad():
            output = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            predictions = torch.sigmoid(output).cpu().numpy()
            predictions = predictions.flatten()
            logging.info(f"Predictions obtained with length {len(predictions)}")
        
        threshold = 0.5
        predictions_label = [1 if p >= threshold else 0 for p in predictions]
        df['Predictions'] = predictions
        df['Label_Prediction'] = predictions_label

        # Include the reasons for risk scores
        df['Risk_Reasons'] = processor.df['Risk_Reasons']

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Fuzzy_Results_{timestamp}.csv"
        filepath = os.path.join(output_directory, filename)
        df.to_csv(filepath, index=False)
        logging.info(f"Results saved to {filepath}")

        return df['Label'].tolist(), predictions, df
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        return None, None, None

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('hybrid_gcn_lstm_model.pth', map_location=device)
    model_params = {k: v for k, v in checkpoint['hyperparameters'].items() if k != 'lr'}
    model = EdgeGCN_LSTM(**model_params)
    model.to(device)
    model.eval()

    output_directory = 'Hybrid_GCN_LSTM'
    os.makedirs(output_directory, exist_ok=True)

    msg_queue = queue.Queue(maxsize=100)
    consumer_thread = threading.Thread(target=consumer_thread, args=(msg_queue,))
    processing_thread = threading.Thread(target=processing_thread, args=(msg_queue, model, device, output_directory))

    consumer_thread.start()
    processing_thread.start()

    consumer_thread.join()
    processing_thread.join()

