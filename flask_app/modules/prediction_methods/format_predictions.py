import pandas as pd

def format_predictions(start, end, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel, time_step, target_row, results_header, y_column_flag, config):    
    # Configure date frequency and offset based on datelevel and endDateTime presence
    if datelevel == 'hour':
        freq = 'H'
        offset = pd.DateOffset(hours=int(time_step))
    elif datelevel == 'day':
        freq = 'D'
        offset = pd.DateOffset(days=int(time_step))
        end = end.replace(hour=0)
    elif datelevel == 'month':
        freq = 'MS'
        offset = pd.DateOffset(months=int(time_step))
        end = end.replace(hour=0)
    elif datelevel == 'year':
        freq = 'YS'
        offset = pd.DateOffset(years=int(time_step))
        end = end.replace(hour=0)
    else:
        raise ValueError("Invalid datelevel")
    
    end += offset
    timestamp = pd.date_range(end=end, periods=len_y_pred_list, freq=freq)
    aggregated_data = pd.DataFrame({'timestamp': timestamp})

    # Extract building name and file from target_row using results_header
    bldgname = target_row[results_header.index('bldgname')]
    building_file = target_row[results_header.index('building_file')]
    aggregated_data['bldgname'] = bldgname
    aggregated_data['building_file'] = building_file

    # Populate columns based on y_pred_lists and y_column_mapping
    for y_column, building, y_pred_list in y_pred_lists:
        if building == building_file.replace('.csv', ''):
            for column in y_column_mapping:
                if column == y_column:
                    aggregated_data[y_column_mapping[column]] = y_pred_list
                    # Calculate CO2 emissions if relevant
                    if column == 'present_elec_kwh':
                        co2_column = y_column_mapping['present_co2_tonh']
                        aggregated_data[co2_column] = aggregated_data[y_column_mapping['present_elec_kwh']] * config['CO2EmissionFactor']

    # Add missing columns for non-present data if flag is 'all'
    if y_column_flag == 'all':
        for column in y_column_mapping.values():
            if column not in aggregated_data.columns:
                aggregated_data[column] = None

    return aggregated_data.reset_index(drop=True)
