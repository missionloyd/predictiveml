import pandas as pd

def format_predictions(start, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel):
    
    # Assuming datelevel is a string representing the desired level of grouping: 'hour', 'day', 'month', or 'year'
    if datelevel == 'hour':
        freq = 'H'
        offset = pd.DateOffset(hours=0)
    elif datelevel == 'day':
        freq = 'D'
        offset = pd.DateOffset(days=0)
    elif datelevel == 'month':
        freq = 'M'
        offset = pd.offsets.MonthBegin(0)
    elif datelevel == 'year':
        freq = 'Y'
        offset = pd.offsets.YearBegin(0)
    else:
        raise ValueError("Invalid datelevel")

    timestamp = pd.date_range(start=start, periods=len_y_pred_list, freq=freq) + offset
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