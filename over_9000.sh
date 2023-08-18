#!/bin/bash

# Function to increment timestamp by 48 hours
increment_timestamp() {
    current_timestamp=$(date -d "$1" +%s)
    next_timestamp=$(date -d "@$((current_timestamp + 48 * 3600))" +"%Y-%m-%dT%H:%M:%S")
}

# Set the start and end dates
start_date="2022-02-04T23:00:00"
end_date="2023-02-04T23:00:00"

# Initialize the current timestamp with the start date
current_timestamp=$start_date

# Create a file to keep a record of the current end_date in the loop
echo "Current end_date in the loop:" > end_date_record.txt

# Loop until the current timestamp is less than the end date
while [[ "$current_timestamp" < "$end_date" ]]; do
    echo "Running script with --endDateTime $current_timestamp"

    # Call your python script here passing the current timestamp
    python3 main.py --prune --run_all --temperature 0.5 --time_step 48 --datelevel hour --endDateTime "$current_timestamp"
    python3 main.py --save_predictions --time_step 48 --datelevel hour --endDateTime "$current_timestamp"

    # Save the current end_date in the loop to the file
    echo "$current_timestamp" >> end_date_record.txt

    # Increment the timestamp by 48 hours
    increment_timestamp "$current_timestamp"
    current_timestamp=$next_timestamp
done