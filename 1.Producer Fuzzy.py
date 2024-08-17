import pandas as pd
from kafka import KafkaProducer
import json

# Initialize the Kafka producer
producer = KafkaProducer(bootstrap_servers=['m3-login3.massive.org.au:9092'],
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))

def send_data_with_count_window(filename, batch_size=100, window_size=500):
    # Read the CSV into a DataFrame
    df = pd.read_csv(filename)
    
    # Sort the DataFrame by the 'Time_step' column
    df = df.sort_values(by='Time_step')
    
    # Initialize an empty list to hold the current window of records
    window = []
    
    # Iterate over the DataFrame row by row
    total_batches = (len(df) + batch_size - 1) // batch_size
    current_window_size = 0
    window_number = 1
    
    for index, row in df.iterrows():
        # Add the row to the current window
        window.append(row.to_dict())
        current_window_size += 1
        
        # If the window is full, send it as a batch to Kafka
        if current_window_size >= window_size:
            try:
                producer.send('kraft-test', value=window)
                producer.flush()  # Ensure all messages are sent
                print(f"Successfully sent window {window_number} with {current_window_size} records")
            except Exception as e:
                print(f"Error sending window {window_number}: {str(e)}")  # Error handling
            
            # Clear the window and reset the counter
            window.clear()
            current_window_size = 0
            window_number += 1
    
    # Send any remaining records in the window (if not empty)
    if window:
        try:
            producer.send('kraft-test', value=window)
            producer.flush()  # Ensure all messages are sent
            print(f"Successfully sent final window with {current_window_size} records")
        except Exception as e:
            print(f"Error sending final window: {str(e)}")  # Error handling

# Send data with count-based windowing
send_data_with_count_window('Thesis/test.csv', batch_size=100, window_size=500)
