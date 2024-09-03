from kafka import KafkaProducer
import json
import pandas as pd
import time

# Initialize the Kafka producer with enhanced reliability configurations
producer = KafkaProducer(
    bootstrap_servers=['m3-login3.massive.org.au:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    acks='all',  # Wait for acknowledgements from all in-sync replicas
    retries=10,  # Retry up to 10 times
    retry_backoff_ms=100  # Wait 100ms between retries
)

def send_data_with_count_window(filename, batch_size=100, window_size=100000, delay_seconds=2):
    # Read the CSV into a DataFrame
    df = pd.read_csv(filename)
    df = df.sort_values(by='Time_step')
   # df=df.head(100000)
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
                print(f"Error sending window {window_number}: {str(e)}")
            
            # Clear the window and reset the counter
            window.clear()
            current_window_size = 0
            window_number += 1
            
            # Add a delay between sending batches
            time.sleep(delay_seconds)
    
    # Send any remaining records in the window (if not empty)
    if window:
        try:
            producer.send('kraft-test', value=window)
            producer.flush()  # Ensure all messages are sent
            print(f"Successfully sent final window with {current_window_size} records")
        except Exception as e:
            print(f"Error sending final window: {str(e)}")

send_data_with_count_window('Thesis/test.csv', batch_size=100, window_size=500)
