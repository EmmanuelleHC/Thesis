import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from scipy.stats import ks_2samp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import os
import csv
from collections import deque
import time
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
import json
from torch.utils.tensorboard import SummaryWriter  # TensorBoard import

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up logging
logging.basicConfig(
    filename='Log/consumer_risk_based.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Starting the script.")

# Kafka consumer creation
def create_consumer():
    try:
        consumer = KafkaConsumer(
            'kraft-test',
            bootstrap_servers=['m3-login3.massive.org.au:9092'],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            fetch_max_wait_ms=4000,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        logging.info("Kafka consumer created successfully.")
        return consumer
    except Exception as e:
        logging.error(f"Error creating Kafka consumer: {e}")
        raise

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
      #  self.df['Time_step'] = pd.to_datetime(self.df['Time_step'])
     #   self.df = self.df.sort_values(by=['Sender_Customer_Id', 'Time_step'])

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
def store_metrics(metrics, metrics_file, writer, batch_number):
    try:
        logging.info(f"Storing metrics for batch number {batch_number}.")
        
        # Write metrics to TensorBoard
        writer.add_scalar('F1_Score', metrics['f1_score'], batch_number)
        writer.add_scalar('Precision', metrics['precision'], batch_number)
        writer.add_scalar('Recall', metrics['recall'], batch_number)
        writer.add_scalar('AUC', metrics['auc'], batch_number)
        writer.add_scalar('PR_AUC', metrics['pr_auc'], batch_number)
        writer.add_scalar('Average_F1', metrics['average_f1'], batch_number)

        # Log confusion matrix to TensorBoard
        cm = metrics['confusion_matrix']
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for Batch {batch_number}')
        plt.colorbar()
        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        writer.add_figure(f'Confusion_Matrix/Batch_{batch_number}', figure)
        plt.close(figure)

        logging.info(f"Metrics stored successfully for batch number {batch_number}.")

        # Write metrics to CSV
        file_exists = os.path.isfile(metrics_file)
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()  # Write header if file doesn't exist
            writer.writerow(metrics)
        logging.info(f"Metrics successfully written to {metrics_file}")

    except Exception as e:
        logging.error(f"Error storing metrics: {e}")
        raise


def store_results(results, results_file):
    try:
        logging.info(f"Storing results to {results_file}")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        # Store results in a separate file for each batch
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"Results stored successfully in {results_file}.")
    except Exception as e:
        logging.error(f"Error storing results as CSV: {e}")
        raise

def process_batch(sliding_window, true_labels, results, all_f1_scores, batch_number, metrics_file, output_folder, writer):
    try:
        logging.info(f"Processing batch number {batch_number}.")

        # Calculate metrics
        f1 = f1_score(true_labels, sliding_window)
        precision = precision_score(true_labels, sliding_window)
        recall = recall_score(true_labels, sliding_window)
        conf_matrix = confusion_matrix(true_labels, sliding_window).tolist()
        all_f1_scores.append(f1)  # Add the F1 score for this window to the list

        fpr, tpr, _ = roc_curve(true_labels, sliding_window)
        roc_auc = auc(fpr, tpr)

        # Calculate Precision-Recall Curve and AUC
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, sliding_window)
        pr_auc = auc(recall_curve, precision_curve)
        metrics = {
            "batch_number": batch_number,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "count": len(results),  # Number of processed transactions in this batch
            "last_f1": f1,  # Last F1 score for this window
            "average_f1": np.mean(all_f1_scores) if all_f1_scores else None,  # Average F1 over all windows,
            "auc": roc_auc,  # Updated key to match the original CSV storage key
            "pr_auc": pr_auc  # Store AUC for PR curve
        }

        # Store the metrics and results as CSV
        store_metrics(metrics, metrics_file, writer, batch_number)
        logging.info(f"Metrics stored for batch number {batch_number}.")

        # Define results file path for the current batch
        results_file = os.path.join(output_folder, f'results_batch_{batch_number}.csv')
        store_results(results, results_file)
        logging.info(f"Results stored for batch number {batch_number}.")

    except Exception as e:
        logging.error(f"Error during batch processing: {e}")
        raise

def consume_and_evaluate(model_path='risk_based.pth', window_size=100000, output_folder="Risk_Based", timeout=1800):
    logging.info("Starting the consumption and evaluation process.")
    consumer = create_consumer()
    sliding_window = deque(maxlen=window_size)
    true_labels = deque(maxlen=window_size)
    all_f1_scores = []  # To store all F1 scores for average-F1 calculation
    results = []
    count = 0
    batch_number = 0
    last_message_time = time.time()

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path).to(device)
    model.eval()  # Set model to evaluation mode

    # Define metrics file path
    metrics_file = os.path.join(output_folder, 'metrics.csv')

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_folder, 'tensorboard_logs'))

    try:
        for message in consumer:
            transactions = message.value  # Assuming message.value is a list of transactions

            # Update last message time
            last_message_time = time.time()

            logging.info(f"Received message with {len(transactions)} transactions.")

            if not isinstance(transactions, list) or len(transactions) == 0:
                logging.warning("Received an empty or non-list message. Skipping.")
                continue

            batch_transactions = []

            for transaction in transactions:
                if not isinstance(transaction, dict):
                    logging.warning("Received data that is not a dictionary. Skipping.")
                    continue
                batch_transactions.append(transaction)

            # Process the entire batch at once
            graph_data_processor = GraphDataProcessor(pd.DataFrame(batch_transactions))
            data = graph_data_processor.prepare_graph_data().to(device)

            # Get the model's prediction
            with torch.no_grad():
                output = model(data.x, data.edge_index, data.edge_attr)
                predictions = (torch.sigmoid(output) > 0.5).cpu().numpy().astype(int)

            # Update sliding windows and results with the batch predictions
            sliding_window.extend(predictions)
            true_labels.extend(data.y.cpu().numpy())
            for transaction, prediction in zip(batch_transactions, predictions):
                results.append({
                    "Transaction_ID": transaction.get('Transaction_Id', 'Unknown'),  # Use 'Unknown' if ID is not available
                    "Transaction_Type": transaction.get('Transaction_Type', 'Unknown'),  # Use 'Unknown' if type is not available
                    "Predicted_Label": prediction,
                    "True_Label": transaction.get('Label', 0)
                })

                count += 1  # Increment by 1 for each transaction processed

            logging.info(f"Processed {len(transactions)} transactions, total processed: {count}")

            # Check if the window size is reached or timeout has occurred
            if count % window_size == 0 or (time.time() - last_message_time > timeout):
                logging.info(f"Processing batch number {batch_number}.")
                process_batch(sliding_window, true_labels, results, all_f1_scores, batch_number, metrics_file, output_folder, writer)
                results = []
                batch_number += 1  # Increment batch number for the next set

            # Reset the last_message_time after processing the batch
            last_message_time = time.time()

        # Final processing of any remaining data (including partial batches)
        if results:  # This ensures any leftover transactions are processed
            logging.info("Final processing of remaining transactions.")
            process_batch(sliding_window, true_labels, results, all_f1_scores, batch_number, metrics_file, output_folder, writer)
            logging.info("Final batch stored.")

    except Exception as e:
        logging.error(f"Error during consumption and evaluation: {e}")
        raise
    finally:
        writer.close()  # Ensure the TensorBoard writer is closed

if __name__ == "__main__":
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        consume_and_evaluate()
    except Exception as e:
        logging.critical(f"Critical error occurred: {e}")
