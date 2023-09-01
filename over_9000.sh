#!/bin/bash

# Function to increment timestamp
increment_timestamp() {
    # current_timestamp=$(date -d "$1" +%s)
    # 1. 24 hours / 1 Day
    input_date="$1"
    current_timestamp=$(date -d "$input_date" +%s)
    export TZ="UTC"
    next_timestamp=$(date -d "@$((current_timestamp + 1 * 24 * 3600))" +"%Y-%m-%dT23:00:00")

    # 3. 1 month
    # current_month=$(date -d "$1" +"%-m")  # Remove leading zero from month
    # next_month=$((current_month + 1))
    # next_year=$(date -d "$1" +"%Y")

    # if [ "$next_month" -gt 12 ]; then
    #     next_month=1
    #     next_year=$((next_year + 1))
    # fi

    # last_day_of_next_month=$(date -d "$next_year-$next_month-1 +1 month -1 day" "+%-d")
    # next_timestamp=$(date -d "$next_year-$next_month-$last_day_of_next_month 23:00:00" +"%Y-%m-%dT%H:%M:%S")
    
    #4. 1 year
    # next_timestamp=$(date -d "@$((current_timestamp + 365 * 24 * 3600))" +"%Y-%m-%dT%H:%M:%S")
}

# Set the start and end dates
# 1. Hour / Day
start_date="2022-01-31T23:00:00"
end_date="2023-02-04T23:00:00"
# 3. Month
# start_date="2022-01-31T23:00:00"
# end_date="2023-01-31T23:00:00"
# 3.1 12 Month
# start_date="2023-01-31T23:00:00"
# end_date="2024-01-31T23:00:00"
# 4. Year
# start_date="2023-12-31T23:00:00"
# end_date="2024-12-31T23:00:00"

# start_date="2023-02-04T23:00:00"
# end_date="2023-08-26T23:00:00"

# Initialize the current timestamp with the start date
current_timestamp=$start_date

# Create a file to keep a record of the current end_date in the loop
echo "Current end_date in the loop:" > end_date_record.txt

# Loop until the current timestamp is less than the end date
while [[ "$current_timestamp" < "$end_date" ]]; do
    echo "Running script with --endDateTime $current_timestamp"

    # Extract year and month from the current timestamp
    year=$(date -u -d "$current_timestamp" +"%Y")
    month=$(date -u -d "$current_timestamp" +"%m")

    # Calculate the next month's timestamp
    tmp_timestamp=$(date -u -d "$year-$month-01 + 1 month" +"%Y-%m-%dT%H:%M:%S")

    # Calculate the number of days in the next month
    year_next=$(date -u -d "$tmp_timestamp" +"%Y")
    month_next=$(date -u -d "$tmp_timestamp" +"%m")
    days_in_month=$(date -d "$year_next-$month_next-01 + 1 month - 1 day" +"%d")

    # time_step=$days_in_month
    # echo "$days_in_month"
    time_step=24
    datelevel="hour"
    table="spaces"

    # Call your python script here passing the current timestamp
    python3 main.py --prune --run_all --save_predictions --temperature 0.5 --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${datelevel}.csv" --endDateTime "$current_timestamp"
    # python3 main.py --save_predictions --time_step "$time_step" --datelevel "$datelevel" --results_file "${datelevel}.csv" --endDateTime "$current_timestamp"

    # Save the current end_date in the loop to the file
    echo "$current_timestamp" >> end_date_record.txt

    # Increment the timestamp by 48 hours
    increment_timestamp "$current_timestamp"
    current_timestamp=$next_timestamp
done