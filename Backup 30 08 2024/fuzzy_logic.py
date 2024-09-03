import json
from kafka import KafkaConsumer
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime
import os
import csv
import pandas as pd
# Antecedents (inputs)
cross_border = ctrl.Antecedent(np.arange(0, 2, 1), 'cross_border')
country_risk = ctrl.Antecedent(np.arange(0, 2, 1), 'country_risk')
pep_involvement = ctrl.Antecedent(np.arange(0, 2, 1), 'pep_involvement')
transaction_type = ctrl.Antecedent(np.arange(0, 3, 1), 'transaction_type')  # Adjusted for clarity

# Membership Functions for PEP Involvement (binary)
pep_involvement['no'] = fuzz.trimf(pep_involvement.universe, [0, 0, 0.5])
pep_involvement['yes'] = fuzz.trimf(pep_involvement.universe, [0.5, 1, 1])

# Setup membership functions for other antecedents
cross_border['low'] = fuzz.trimf(cross_border.universe, [0, 0, 1])
cross_border['high'] = fuzz.trimf(cross_border.universe, [0, 1, 1])
country_risk['low'] = fuzz.trimf(country_risk.universe, [0, 0, 1])
country_risk['high'] = fuzz.trimf(country_risk.universe, [0, 1, 1])
transaction_type['crypto_transfer'] = fuzz.trimf(transaction_type.universe, [0, 0, 1])
transaction_type['payment'] = fuzz.trimf(transaction_type.universe, [1, 1, 2])
transaction_type['other'] = fuzz.trimf(transaction_type.universe, [2, 2, 2])

# Consequent (output)
risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')
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
def evaluate_transaction(row):
    if row['Transaction_Type'] in ['CRYPTO-TRANSFER', 'PAYMENT']:
        transaction_type_value = 1
    else:
        transaction_type_value = 0

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
    reasons = []

    # Evaluate the reasons for the risk score
    if transaction_type_value == 1:
        reasons.append("High-risk transaction type (crypto or payment)")
    if pep_involvement_value == 1:
        reasons.append("PEP involvement in transaction")
    if cross_border_value == 1:
        reasons.append("Cross-border transaction")
    if country_risk_value == 1:
        reasons.append("Transaction involves high-risk country")
    risk_score = aml_sim.output['risk']

    return risk_score,reasons

