import os
import csv
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kafka import KafkaConsumer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc,precision_recall_curve
from collections import deque
import json
import logging
import time
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    filename='Log/consumer_rule_based.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Starting the script.")

# Define fuzzy logic system
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

logging.info("Fuzzy logic system defined.")

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

def evaluate_transaction(transaction):
    try:
        logging.info("Evaluating transaction.")
        
        # Check if transaction is a list and get the first item if so
        if isinstance(transaction, list):
            if len(transaction) == 0:
                raise ValueError("Received an empty list for transaction data")
            transaction = transaction[0]  # Assuming you want the first item in the list

        # Map Transaction_Type to a fuzzy value
        transaction_type_value = {
            'CRYPTO-TRANSFER': 0,
            'PAYMENT': 1,
            'OTHER': 2
        }.get(transaction.get('Transaction_Type', 'OTHER'), 2)  # Default to 'OTHER'

        pep_involvement_value = 1 if transaction.get('Bene_Is_Pep', False) or transaction.get('Sender_Is_Pep', False) else 0
        cross_border_value = 1 if transaction.get('Sender_Country', '') != transaction.get('Bene_Country', '') else 0
        high_risk_countries = ['Iran', 'Syria', 'North-Korea']
        country_risk_value = 1 if transaction.get('Bene_Country', '') in high_risk_countries else 0

        aml_sim.input['transaction_type'] = transaction_type_value
        aml_sim.input['pep_involvement'] = pep_involvement_value
        aml_sim.input['cross_border'] = cross_border_value
        aml_sim.input['country_risk'] = country_risk_value

        aml_sim.compute()
        risk_score = aml_sim.output['risk']
        label = 1 if risk_score >= 60 else 0
        return label, risk_score
    except Exception as e:
        logging.error(f"Error evaluating transaction: {e}")
        raise

def store_metrics(metrics, metrics_file):
    try:
        logging.info(f"Storing metrics to {metrics_file}")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        # Store metrics in a single file
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Write header if file is empty
                writer.writerow(['Batch_Number', 'F1_Score', 'Precision', 'Recall', 'Confusion_Matrix', 'Count', 'Last_F1', 'Average_F1', 'AUC'])
            writer.writerow([
                metrics['batch_number'], metrics['f1_score'], metrics['precision'], metrics['recall'], 
                metrics['confusion_matrix'], metrics['count'], metrics['last_f1'], metrics['average_f1'], 
                metrics['auc']
            ])
        logging.info("Metrics stored successfully.")
    except Exception as e:
        logging.error(f"Error storing metrics as CSV: {e}")
        raise

# Function to store results in separate CSV files for each batch
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

def process_batch(sliding_window, true_labels, risk_scores, results, all_f1_scores, batch_number, metrics_file, output_folder):
    try:
        logging.info(f"Processing batch number {batch_number}.")
        
        # Calculate metrics
        f1 = f1_score(true_labels, sliding_window)
        precision = precision_score(true_labels, sliding_window)
        recall = recall_score(true_labels, sliding_window)
        conf_matrix = confusion_matrix(true_labels, sliding_window).tolist()

        all_f1_scores.append(f1)  # Add the F1 score for this window to the list

        # Calculate AUC for ROC
        fpr, tpr, _ = roc_curve(true_labels, risk_scores)
        roc_auc = auc(fpr, tpr)

        # Calculate Precision-Recall Curve and AUC
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, risk_scores)
        pr_auc = auc(recall_curve, precision_curve)
        metrics = {
            "batch_number": batch_number,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "count": len(results),  # Number of processed transactions in this batch
            "last_f1": f1,  # Last F1 score for this window
            "average_f1": np.mean(all_f1_scores) if all_f1_scores else None,  # Average F1 over all windows
            "auc": roc_auc,  # Updated key to match the original CSV storage key
            "pr_auc": pr_auc  # Store AUC for PR curve
        }
        
        # Store the metrics and results as CSV
        store_metrics(metrics, metrics_file)
        logging.info(f"Metrics stored for batch number {batch_number}.")

        # Define results file path for the current batch
        results_file = os.path.join(output_folder, f'results_batch_{batch_number}.csv')
        store_results(results, results_file)
        logging.info(f"Results stored for batch number {batch_number}.")

        # Plot and save ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_file = os.path.join(output_folder, f'roc_curve_batch_{batch_number}.png')
        plt.savefig(roc_file)
        plt.close()
        logging.info(f"ROC curve saved for batch number {batch_number}.")

        # Plot and save Precision-Recall curve
        plt.figure()
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        pr_file = os.path.join(output_folder, f'pr_curve_batch_{batch_number}.png')
        plt.savefig(pr_file)
        plt.close()
        logging.info(f"PR curve saved for batch number {batch_number}.")
    except Exception as e:
        logging.error(f"Error during batch processing: {e}")
        raise
def consume_and_evaluate(window_size=100000, output_folder="Rule_Based", timeout=1800):
    logging.info("Starting the consumption and evaluation process.")
    consumer = create_consumer()
    sliding_window = deque(maxlen=window_size)
    true_labels = deque(maxlen=window_size)
    risk_scores = deque(maxlen=window_size)
    all_f1_scores = []  # To store all F1 scores for average-F1 calculation
    results = []
    count = 0
    batch_number = 0
    last_message_time = time.time()

    # Define metrics file path
    metrics_file = os.path.join(output_folder, 'metrics.csv')

    try:
        for message in consumer:
            transactions = message.value  # Assuming message.value is a list of transactions

            # Update last message time
            last_message_time = time.time()

            logging.info(f"Received message with {len(transactions)} transactions.")

            if not isinstance(transactions, list) or len(transactions) == 0:
                logging.warning("Received an empty or non-list message. Skipping.")
                continue

            for transaction in transactions:
                if not isinstance(transaction, dict):
                    logging.warning("Received data that is not a dictionary. Skipping.")
                    continue

                label, risk_score = evaluate_transaction(transaction)
                true_label = transaction.get('Label', 0)  # Assuming the true label is provided in the message

                sliding_window.append(label)
                true_labels.append(true_label)
                risk_scores.append(risk_score)
                results.append({
                    "Transaction_ID": transaction.get('Transaction_Id', 'Unknown'),  # Use 'Unknown' if ID is not available
                    "Transaction_Type": transaction.get('Transaction_Type', 'Unknown'),  # Use 'UNKNOWN' if type is not available
                    "Predicted_Label": label,
                    "Risk_Score": risk_score,
                    "True_Label": true_label
                })

                count += 1  # Increment by 1 for each transaction processed

            logging.info(f"Processed {len(transactions)} transactions, total processed: {count}")

            # Check if the window size is reached or timeout has occurred
            if count % window_size == 0 or (time.time() - last_message_time > timeout):
                logging.info(f"Masuk sini")
                process_batch(sliding_window, true_labels, risk_scores, results, all_f1_scores, batch_number, metrics_file, output_folder)
                results = []
                batch_number += 1  # Increment batch number for the next set

            # Reset the last_message_time after processing the batch
            last_message_time = time.time()

        # Final processing of any remaining data (including partial batches)
        if results:  # This ensures any leftover transactions are processed
            logging.info("Final processing of remaining transactions.")
            process_batch(sliding_window, true_labels, risk_scores, results, all_f1_scores, batch_number, metrics_file, output_folder)
            logging.info("Final batch stored.")

    except Exception as e:
        logging.error(f"Error during consumption and evaluation: {e}")
        raise


if __name__ == "__main__":
    try:
        consume_and_evaluate()
    except Exception as e:
        logging.critical(f"Critical error occurred: {e}")
