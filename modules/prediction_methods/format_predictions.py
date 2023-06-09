import pandas as pd

def format_predictions(start, end, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel, time_step):
    # Assuming datelevel is a string representing the desired level of grouping: 'hour', 'day', 'month', or 'year'
    if datelevel == 'hour':
        freq = 'H'
        offset = pd.DateOffset(hours=int(time_step))
    elif datelevel == 'day':
        freq = 'D'
        offset = pd.DateOffset(days=int(time_step))
    elif datelevel == 'month':
        freq = 'MS'
        offset = pd.offsets.MonthBegin(int(time_step))
    elif datelevel == 'year':
        freq = 'YS'
        offset = pd.offsets.YearBegin(int(time_step))
    else:
        raise ValueError("Invalid datelevel")
    
    end = end + offset  # Add the offset to the end date
    timestamp = pd.date_range(end=end, periods=len_y_pred_list, freq=freq)
    aggregated_data = pd.DataFrame({'timestamp': timestamp})

    for y_column, y_pred_list in y_pred_lists:
        for column in y_column_mapping:
            if column == y_column:
                aggregated_data[y_column_mapping[column]] = y_pred_list
            # else:
            #     aggregated_data[y_column_mapping[column]] = None

    # Add the missing columns after resampling
    for column in y_column_mapping.values():
        if column not in aggregated_data.columns:
            aggregated_data[column] = None

    aggregated_data = aggregated_data.reset_index(drop=True)

    # return results
    return aggregated_data