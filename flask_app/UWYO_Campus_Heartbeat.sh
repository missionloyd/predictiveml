#!/bin/bash
export TZ="UTC"  # Set timezone to UTC

# Function to increment timestamp
increment_timestamp() {
    # current_timestamp=$(date -d "$1" +%s)
    # 1. 24 hours / 1 Day
    input_date="$1"

    # Get the start of the input day
    start_of_day=$(date -d "$input_date" +"%Y-%m-%d 00:00:00")
    current_timestamp=$(date -d "$start_of_day" +%s)

    # Add one day's worth of seconds
    next_timestamp=$(date -d "@$((current_timestamp + 24 * 3600))" +"%Y-%m-%dT23:00:00")
}

# Set the start and end dates
# 1. Hour / Day
start_date="2021-12-31T23:00:00"
end_date="2023-02-04T23:00:00"
end_date_record_file="end_date_record.txt"

# start_date="2023-08-01T23:00:00"
# end_date="2023-08-26T23:00:00"

# Initialize the current timestamp with the start date
current_timestamp=$start_date

# Create a file to keep a record of the current end_date in the loop
echo "Current end_date in the loop:" > $end_date_record_file

# Loop until the current timestamp is less than the end date
while [[ "$current_timestamp" < "$end_date" ]]; do
    echo "Running script with --endDateTime $current_timestamp"

    # Extract year and month from the current timestamp
    year=$(date -u -d "$current_timestamp" +"%Y")
    month=$(date -u -d "$current_timestamp" +"%m")
    day=$(date -u -d "$current_timestamp" +"%d")
    last_day_of_month=$(date -d "${year}-${month}-01 +1 month -1 day" +%d)
    last_day_of_year=$(date -d "${year}-12-31" +%j)

    table="spaces"

    time_step=24
    datelevel="hour"
    python3 main.py --model_type xgboost --temperature 0.5 --prune --run_all --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv" --endDateTime "$current_timestamp"
    
    time_step=1
    datelevel="day"
    python3 main.py --model_type xgboost --temperature 0.5 --prune --run_all --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv" --endDateTime "$current_timestamp"


    if [[ "$day" == "$last_day_of_month" ]]; then
        time_step=1
        datelevel="month"
        python3 main.py --prune --run_all --temperature 1.0 --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv" --endDateTime "$current_timestamp"
    fi

    if [[ "$day" == "$last_day_of_year" ]]; then
        time_step=1
        datelevel="year"
        python3 main.py --prune --run_all --temperature 1.0 --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv" --endDateTime "$current_timestamp"
    fi
    
    # Save the current end_date in the loop to the file
    echo "$current_timestamp" >> $end_date_record_file

    # Increment the timestamp by 48 hours
    increment_timestamp "$current_timestamp"
    current_timestamp=$next_timestamp
done

time_step=24
datelevel="hour"
python3 main.py --prune --run_all --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv"

time_step=7
datelevel="day"
python3 main.py --prune --run_all --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv"

time_step=12
datelevel="month"
python3 main.py --prune --run_all --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv"

time_step=1
datelevel="year"
python3 main.py --prune --run_all --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv"
