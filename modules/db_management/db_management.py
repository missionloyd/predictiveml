# https://towardsdatascience.com/how-to-build-a-relational-database-from-csv-files-using-python-and-heroku-20ea89a55c63
import psycopg2
import pandas as pd
import os, sys
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# python3 create_tables.py
# python3 populate_tables.py

def run_syntax(db_connection: psycopg2, syntax: str, entry: tuple) -> None:
    """
    Run syntax.

    :param db_connection: Database connection object.
    :param syntax: Syntax for execution.
    """
    # print(syntax, entry)
    cur = db_connection.cursor()

    statement = syntax
    
    if entry:
        statement = cur.mogrify(syntax, entry)
        
    # print(entry)
    cur.execute(statement)
    cur.close()


def create_table(schema: str, table: str) -> None:
    """
    Create a table in the DB based on a schema.

    :param schema: The table schema.

    :param schema: The schema.
    :param table: The name of the table.
    """
    db_connection = psycopg2.connect(
        host=os.environ["hostname"],
        user=os.environ["user"],
        password=os.environ["password"],
        dbname=os.environ["database"],
    )

    # Create table if it does not yet exist
    run_syntax(db_connection=db_connection, syntax=f"CREATE TABLE IF NOT EXISTS {table}({schema})", entry=tuple())

    db_connection.commit()
    db_connection.close()


def populate_table(table_name: str, df: pd.DataFrame) -> None:
    """
    Populate a table in the database from a pandas dataframe.

    :param table_name: The name of the table in the DB that we will add the values in df to.
    :param df: The dataframe that we use for puplating the table.
    """
    db_connection = psycopg2.connect(
        host=os.environ["hostname"],
        user=os.environ["user"],
        password=os.environ["password"],
        dbname=os.environ["database"],
    )

    # Check that all columns are present in the CSV file
    cur = db_connection.cursor()
    cur.execute(f"SELECT * FROM {table_name} LIMIT 0")
    cur.close()

    col_names = [i[0] for i in cur.description]
    # df["row_timestamp"] = [datetime.now().strftime("%m-%d-%Y %H:%M:%S")] * len(df.index)

    missing_columns = set(col_names).difference(df.columns)
    assert not missing_columns, f"The following columns are missing in your CSV file: {','.join(missing_columns)}"

    # Re-order CSV
    df = df[col_names]

    # NaN -> NULL
    # df = df.where(pd.notnull(df), None)
    value_placeholders = "("

    for i in (range(0, len(col_names) - 1)):
        value_placeholders += "%s, "

    value_placeholders += "%s)"


    # https://stackoverflow.com/questions/55922003/update-multiple-columns-on-conflict-postgres
    excluded_placeholders = "("
    excluded_col_names = "("
    str_col_names = "("

    for i in (range(0, len(col_names) - 1)):
        str_col_names += str(col_names[i]) + ", "

        if col_names[i] != 'ts' and col_names[i] != 'bldgname':
            excluded_placeholders += "EXCLUDED." + str(col_names[i]) + ", "
            excluded_col_names += str(col_names[i]) + ", "

    excluded_placeholders += "EXCLUDED." + str(col_names[len(col_names)-1]) + ")"
    str_col_names += str(col_names[len(col_names)-1]) + ")"
    excluded_col_names += str(col_names[len(col_names)-1]) + ")"

    # Inject data
    for index, row in df.iterrows():

        new_row = row.where(pd.notnull(row), None)
        run_syntax(db_connection=db_connection, syntax=f"INSERT INTO {table_name} VALUES {value_placeholders} ON CONFLICT(bldgname, ts) DO UPDATE SET {excluded_col_names} = {excluded_placeholders};", entry=tuple(new_row.values))

    db_connection.commit()
    db_connection.close()


    # Original Inject data
    # for index, row in df.iterrows():
    #     new_row = row.where(pd.notnull(row), None) 
    #     run_syntax(db_connection=db_connection, syntax=f"INSERT INTO {table_name} VALUES {value_placeholders};", entry=tuple(new_row.values))
    #     run_syntax(db_connection=db_connection, syntax=f"INSERT INTO {table_name} VALUES{tuple(new_row.values)}" , entry=tuple())
    