#!/bin/bash

# Start Zookeeper
kafka_2.13-3.0.0/bin/zookeeper-server-start.sh kafka_2.13-3.0.0/config/zookeeper.properties &
# Wait for Zookeeper to start up completely (you may adjust the sleep time if needed)
sleep 10

# Start Kafka server
kafka_2.13-3.0.0/bin/kafka-server-start.sh kafka_2.13-3.0.0/config/server.properties &
# Wait for Kafka server to start up completely
sleep 10

# Submit the consumer job
sbatch consumer_hybrid_gcn_lstm.script
# Wait for 3 minutes before submitting the producer job
sleep 180

# Submit the producer job
sbatch producer_gcn_lstm.script

