import json
import os
import threading
import queue
from datetime import datetime
from kafka import KafkaConsumer
from fuzzy_logic import evaluate_transaction
import time
import torch
from torch_geometric.data import Data
import pandas as pd
import shutil
from gcn_lstm_model import EdgeGCN_LSTM, GraphDataProcessor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import ks_2samp
import logging
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def processing_thread(msg_queue, model, device, output_directory, metrics_interval=100):
    results = []
    true_labels = []
    predictions = []
    counter = 0
    
    while True:
        message = msg_queue.get()
        if message is None:
            break
        true_label, prediction, result = process_message(message, model, device)
        results.append(result)
        true_labels.append(true_label)
        predictions.append(prediction)
        counter += 1

        if counter >= metrics_interval:
            precision = precision_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)
            auc = roc_auc_score(true_labels, predictions)
            ks_stat, ks_pvalue = ks_2samp(true_labels, predictions)

            # Log and save metrics
            log_and_save_metrics(output_directory, precision, recall, f1, auc, ks_stat, ks_pvalue)
            
            # Reset for the next interval
            results.clear()
            true_labels.clear()
            predictions.clear()
            counter = 0
        
def process_message(message, model, device, output_directory):
    try:
        decoded_message = json.loads(message.decode('utf-8'))
        df = pd.DataFrame(decoded_message)

        for i, record in df.iterrows():
            risk_score, reasons = evaluate_transaction(record)
            df.at[i, 'risk_score'] = risk_score
            df.at[i, 'fuzzy_result'] = ', '.join(reasons)

        processor = GraphDataProcessor(df)
        graph_data = processor.prepare_graph_data().to(device)

        with torch.no_grad():
            output = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            predictions = torch.sigmoid(output).cpu().numpy()

        threshold = 0.5
        predictions_label = [1 if p >= threshold else 0 for p in predictions]
        df['Predictions'] = predictions
        df['Label_Prediction'] = predictions_label

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Fuzzy_Results_{timestamp}.csv"
        filepath = os.path.join(output_directory, filename)
        df.to_csv(filepath, index=False)
        logging.info(f"Results saved to {filepath}")
    except Exception as e:
        logging.error(f"Error processing message: {e}")

def log_and_save_metrics(output_directory, precision, recall, f1, auc, ks_stat, ks_pvalue):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Metrics_{timestamp}.csv"
    filepath = os.path.join(output_directory, filename)
    metrics_data = pd.DataFrame({
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'AUC': [auc],
        'KS_Statistic': [ks_stat],
        'KS_PValue': [ks_pvalue]
    })
    metrics_data.to_csv(filepath, index=False)
    logging.info(f"Metrics saved to {filepath}")
        
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