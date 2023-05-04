# buildings_list = ['Animal_Science_Data.csv', 'Anthropology_Data.csv', 'Arena_Auditorium_Data.csv', 'Arts_and_Sciences_Data.csv', 'Aven_Nelson_Data.csv', 'Berry_Center_Data.csv', 'Beta_House_Data.csv', 'Biological_Sciences_Data.csv', 'Centennial_Complex_Data.csv', 'Central_Energy_Plant_Data.csv', 'Centrex_Data.csv', 'Cheney_Center_Data.csv', 'Child_Care_Data.csv', 'Classroom_Building_Data.csv', 'Coe_Library_Data.csv', 'College_of_Agriculture_Data.csv', 'College_of_Business_Data.csv', 'College_of_Education_Data.csv', 'College_of_Law_Data.csv', 'Corbett_Data.csv', 'EERB_Data.csv', 'Earth_Sciences_Data.csv', 'Education_Annex_Data.csv', 'Energy_Innovation_Center_Data.csv', 'Engineering_Data.csv', 'Enzi-STEM_Data.csv', 'Fieldhouse_Data.csv', 'Fieldhouse_North_Data.csv', 'Fine_Arts_Data.csv', 'Gateway_Center_Data.csv', 'General_Storage_Data.csv', 'Geological_Survey_Data.csv', 'Geology_Data.csv', 'HAPC_and_RAC_Data.csv', 'Half_Acre_Gym_Data.csv', 'Health_Science_and_Pharmacy_Data.csv', 'High_Bay_Data.csv', 'Hoyt_Hall_Data.csv', 'Indoor_Practice_Facility_Data.csv', 'Information_Technology_Data.csv', 'Knight_Hall_Data.csv', 'McWhinnie_Hall_Data.csv', 'Merica_Hall_Data.csv', 'Old_Main_Data.csv', 'Physical_Science_Data.csv', 'RMMC_Data.csv', 'Ross_Hall_Data.csv', 'Science_Initiative_Building_Data.csv', 'Service_Building_Data.csv', 'Stadium_Data.csv', 'Student_Union_Data.csv', 'Vet_Lab_Data.csv', 'Visual_Arts_Data.csv', 'WRI_and_Bureau_of_Mines_Data.csv', 'WTBC_Data.csv', 'Williams_Conservatory_Data.csv', 'Wyoming_Hall_Data.csv']

import os, sys
import pandas as pd
import numpy as np
from prophet import Prophet
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

load_dotenv()

print('*** Running predictive analytics ***')
par_folder = '.'

in_path = par_folder + '/clean_data/'

buildings_list = ['Stadium_Data.csv']

for building in buildings_list:
    df = pd.read_csv(in_path + building)

    # Convert the data into a Pandas dataframe
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.drop_duplicates(subset=["bldgname", "ts"])
    df = df.sort_values(["bldgname", "ts"])
    # df = df.set_index("ts")

    # Group the dataframe by building name and timestamp
    groups = df.groupby("bldgname")

    orig_cols = df.columns
    y_columns = ["present_elec_kwh", "present_htwt_mmbtu", "present_wtr_usgal", "present_chll_tonhr", "present_co2_tons"]
    header = ["ts"] + y_columns
    new_header = ["campus", "bldgname", "ts"] + y_columns + [col.replace('present', 'predicted') for col in y_columns]

    # Train the Prophet models on the data
    models = {}
    rmse_scores = {}

    for name, group in groups:
        bldgname = name

        group = group.drop_duplicates(subset=["ts"])
        model_data = group[header]

        for y in y_columns:
            if model_data[y].count() >= 365*24 and y != 'present_co2_tons':
                model_data = model_data.rename(columns={ "ts": "ds", y: "y" })
                model_data = model_data.sort_values(["ds"])

                # create a new future dataframe for this building and y column
                last_entry = model_data["y"].last_valid_index()
 
                if last_entry is not None:
                    last_ts = model_data.loc[last_entry, "ds"]
                    print(bldgname, last_ts)
                    start = last_ts + pd.Timedelta(hours=1)
                    future = pd.DataFrame(pd.date_range(start=start, periods=365*24, freq='H'), columns=["ds"])

                    model = Prophet()
                    model.fit(model_data)
                    models[(bldgname, y)] = (model, future)

                    # compute RMSE
                    train_size = int(len(model_data["y"]) * 0.8)
                    test_size = len(model_data["y"]) - train_size

                    train_data = np.array(model_data["y"])[0:train_size]
                    test_data = np.array(model_data["y"])[train_size:]

                    y_true = test_data
                    y_pred = model.predict(model_data)["yhat"][train_size:].values

                    rmse = np.sqrt(mean_squared_error(y_true[~np.isnan(y_true)], y_pred[~np.isnan(y_true)]))
                    print("RMSE:", rmse)

                    model_data = model_data.rename(columns={ "ds": "ts", "y": y })

    # Generate predictions for future time periods
    # future = pd.DataFrame(pd.date_range(start=df["ts"].max(), periods=365*24, freq='H'), columns=["ds"])
    forecasts = {}
    prediction_data = pd.DataFrame()

    for name, (model, future) in models.items():
        bldgname, y = name
        forecast = model.predict(future)
        forecast = forecast[["ds", "yhat"]]
        forecast = forecast.rename(columns={"yhat": y.replace("present", "predicted")})
        forecasts[(bldgname, y)] = forecast

    # Merge the forecasts with the original dataframe
    for name, forecast in forecasts.items():
        bldgname, y = name
        # Select the relevant columns from the original data
        orig_data = df[df["bldgname"] == bldgname][orig_cols]
        orig_data = orig_data.rename(columns={"ts": "ds"})

        # Merge the original data with the forecasts
        merged_data = pd.merge(orig_data, forecast, on="ds", how="outer")
        merged_data = merged_data.rename(columns={"ds": "ts"})

        # Update missing campus and bldgname values in merged_data with values from df
        merged_data["campus"] = merged_data["campus"].fillna(df.loc[df["bldgname"] == bldgname, "campus"].iloc[0])
        merged_data["bldgname"] = merged_data["bldgname"].fillna(bldgname)

        # Sort the merged data by building name and timestamp
        merged_data = merged_data.sort_values(["ts"])

        # print(merged_data.columns)

        # create the plot     
        for y in y_columns:
            pred_col = y.replace('present', 'predicted')
            if pred_col in merged_data.columns:
                fig, ax = plt.subplots()
                ax.plot(merged_data[y], label='Actual')
                ax.plot(merged_data[pred_col], label='Forecast')

                # set plot title and axis labels
                commodity = pred_col.split('_')[-2]
                ax.set_title(bldgname + ' Consumption')
                ax.set_xlabel('Time (Hours)')
                ax.set_ylabel(y.split('_')[-2] + ' (' + y.split('_')[-1] + ')')

                # add legend
                ax.legend()
                # print(forecast)
                plot_file = 'predictions/' + (bldgname.split('.')[0]).replace(' ', '_') + '_' + commodity

                if(commodity != 'co2'):
                    plt.savefig(par_folder + '/clean_data/' + plot_file + '.png')
                    prediction_data['ts'] = merged_data['ts']
                    prediction_data[pred_col] = merged_data[pred_col]

                # plt.show()
                plt.close(fig)

    merged_data = pd.merge(merged_data, prediction_data, on="ts", how="outer")

    for y in y_columns:
        pred_col = y.replace('present', 'predicted')
        if pred_col not in merged_data.columns:
            merged_data[pred_col] = pd.Series('')

    merged_data = merged_data.reindex(columns=new_header)
    prediction_file = par_folder + '/clean_data/predictions/' + building
    merged_data.to_csv(prediction_file, index=False)




    # Set the variables
    # hostname = os.environ.get("hostname")
    # user = os.environ.get("user")
    # password = os.environ.get("password")
    # database = os.environ.get("database")

    # # Connect to the PostgreSQL database
    # engine = create_engine(f'postgresql://{user}:{password}@{hostname}:5432/{database}')

    # Query the existing data from the spaces table
    # df = pd.read_sql_query('SELECT * FROM spaces', con=engine)

        # Save the merged data to the database
        # merged_data.to_sql("spaces", con=engine, if_exists="replace", index=False)

    # Close the database connection
    # engine.dispose()