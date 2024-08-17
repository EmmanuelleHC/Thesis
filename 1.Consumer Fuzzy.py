import json
from kafka import KafkaConsumer
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime
import os
import csv

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'kraft-test',
    bootstrap_servers=['m3-login3.massive.org.au:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True
)

# Manually seek to the latest offset
consumer.poll(timeout_ms=1000)  # Poll to assign partitions
consumer.seek_to_end()

# Define Antecedents and Consequent
# amount = ctrl.Antecedent(np.arange(0, 10000, 1000), 'amount')
cross_border = ctrl.Antecedent(np.arange(0, 2, 1), 'cross_border')
country_risk = ctrl.Antecedent(np.arange(0, 2, 1), 'country_risk')  # For country risk like Iran, Syria, North Korea
pep_involvement = ctrl.Antecedent(np.arange(0, 2, 1), 'pep_involvement')
transaction_type = ctrl.Antecedent(np.arange(0, 3, 1), 'transaction_type')  # 0: domestic, 1: international, 2: digital

risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# Membership Functions
# amount.automf(3, names=['low', 'medium', 'high'])
cross_border['domestic'] = fuzz.trimf(cross_border.universe, [0, 0, 0.5])
cross_border['international'] = fuzz.trimf(cross_border.universe, [0.5, 1, 1])

country_risk['low'] = fuzz.trimf(country_risk.universe, [0, 0, 0.5])
country_risk['high'] = fuzz.trimf(country_risk.universe, [0.5, 1, 1])

pep_involvement['non_pep'] = fuzz.trimf(pep_involvement.universe, [0, 0, 0.5])
pep_involvement['pep'] = fuzz.trimf(pep_involvement.universe, [0.5, 1, 1])

transaction_type['domestic'] = fuzz.trimf(transaction_type.universe, [0, 0, 0.5])
transaction_type['international'] = fuzz.trimf(transaction_type.universe, [0.5, 1, 1.5])
transaction_type['digital'] = fuzz.trimf(transaction_type.universe, [1.5, 2, 2])

risk['low'] = fuzz.trimf(risk.universe, [0, 0, 50])
risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 80])
risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

# Fuzzy rules
# rule1 = ctrl.Rule(amount['high'] | cross_border['international'], risk['high'])
# rule2 = ctrl.Rule(amount['medium'] & cross_border['domestic'], risk['medium'])
# rule3 = ctrl.Rule(amount['low'] & (cross_border['domestic'] | cross_border['international']), risk['low'])
rule4 = ctrl.Rule(country_risk['high'], risk['high'])
rule5 = ctrl.Rule(pep_involvement['pep'], risk['high'])
rule6 = ctrl.Rule(transaction_type['digital'], risk['medium'])
rule7 = ctrl.Rule(transaction_type['international'], risk['high'])

# Control system setup
aml_control = ctrl.ControlSystem([rule4, rule5, rule6, rule7])
aml_sim = ctrl.ControlSystemSimulation(aml_control)

def evaluate_transaction(row):
    reasons = []

    transaction_type_value = {
        'domestic': 0,
        'international': 1,
        'digital': 2,
        'CRYPTO-TRANSFER': 2,
        'PAYMENT': 2
    }.get(row['Transaction_Type'], 0)

    pep_involvement_value = 1 if row.get('Bene_Is_Pep', False) or row.get('Sender_Is_Pep', False) else 0

    aml_sim.inputs({
        'cross_border': 1 if row['Sender_Country'] != row['Bene_Country'] else 0,
        'country_risk': 1 if row['Bene_Country'] in ['Iran', 'Syria', 'North-Korea'] else 0,
        'pep_involvement': pep_involvement_value,
        'transaction_type': transaction_type_value
    })

    aml_sim.compute()
    risk_score = aml_sim.output['risk']
    if risk_score >= 60:
        if row['Sender_Country'] != row['Bene_Country']:
            reasons.append("Cross Border Transaction")
        if row['Bene_Country'] in ['Iran', 'Syria', 'North-Korea']:
            reasons.append("High Risk Country")
        if pep_involvement_value == 1:
            reasons.append("PEP Involvement")
        if transaction_type_value == 2:
            reasons.append("Digital Transaction")
    return risk_score, ', '.join(reasons)

import time

def process_and_evaluate(timeout_minutes=5):
    start_time = time.time()
    results = []
    while time.time() - start_time < timeout_minutes * 60:
        consumer.poll(timeout_ms=5000)  # Wait up to 5000ms for a new message
        for message in consumer:
            try:
                decoded_message = json.loads(message.value.decode('utf-8'))
                for record in decoded_message:
                    risk_score, reasons = evaluate_transaction(record)
                    record['risk_score'] = risk_score
                    record['fuzzy_result'] = reasons
                    results.append(record)

                if results:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    directory = "Result"
                    os.makedirs(directory, exist_ok=True)
                    filename = f"{directory}/Fuzzy_Results_{timestamp}.csv"
                    with open(filename, mode='w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
                    results.clear()  # Clear results after writing to file

            except Exception as e:
                print(f"Error processing message: {e}")

# Process and evaluate messages from Kafka for 5 minutes
process_and_evaluate()
