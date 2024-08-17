import pandas as pd
from kafka import KafkaProducer
import json

# Initialize the Kafka producer
producer = KafkaProducer(bootstrap_servers=['m3-login3.massive.org.au:9092'],
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))
def send_data(filename, batch_size=100):
    # Read the CSV into a DataFrame
    df = pd.read_csv(filename)
    
    # Sort the DataFrame by the 'Time_step' column
    df = df.sort_values(by='Time_step')
    
    # Count total batches for better logging
    total_batches = (len(df) + batch_size - 1) // batch_size

    # Iterate over the DataFrame in batches
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size].to_dict(orient='records')
        batch_number = start // batch_size + 1  # Calculate current batch number

        try:
            # Send the batch to Kafka
            producer.send('kraft-test', value=batch)
            producer.flush()  # Ensure all messages are sent
            print(f"Successfully sent batch {batch_number} of {total_batches}")
        except Exception as e:
            print(f"Error sending batch {batch_number} from index {start}: {str(e)}")  # Error handling

send_data('Thesis/test.csv')
