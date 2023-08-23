import os
import pandas as pd
import numpy as np
from .db_management import populate_table
from .schema_columns_list import schema_columns_list

def insert_data(merged_list, config):
    table = config['table']

    for df in merged_list:
        print(df)
        # populate_table(table_name=table, df=df)
  
    return