# https://naysan.ca/2020/05/09/pandas-to-postgresql-using-psycopg2-bulk-insert-performance-benchmark/
from io import StringIO
import psycopg2
import psycopg2.extras
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()   


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
        
    # print(statement)
    print(entry)
    cur.execute(statement)
    cur.close()


def create_table(db: str, schema: str, table: str) -> None:
    """
    Create a table in the DB based on a schema.

    :param schema: The table schema.

    :param schema: The schema.
    :param table: The name of the table.
    """
    db_connection = psycopg2.connect(
        host=os.environ[f"{db}-hostname"],
        user=os.environ[f"{db}-user"],
        password=os.environ[f"{db}-password"],
        dbname=os.environ[f"{db}-database"],
    )

    # Create table if it does not yet exist
    run_syntax(db_connection=db_connection, syntax=f"CREATE TABLE IF NOT EXISTS {table}({schema})", entry=tuple())

    db_connection.commit()
    db_connection.close()


def drop_table(db: str, table: str) -> None:
    """
    Drop a table from the database.

    :param table: The name of the table to drop.
    """
    try:
        db_connection = psycopg2.connect(
            host=os.environ[f"{db}-hostname"],
            user=os.environ[f"{db}-user"],
            password=os.environ[f"{db}-password"],
            dbname=os.environ[f"{db}-database"],
        )
        cursor = db_connection.cursor()

        # Drop table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

        db_connection.commit()
        print(f"Table '{table}' dropped successfully.")
    except psycopg2.Error as e:
        print("Error dropping table:", e)
    finally:
        if db_connection:
            db_connection.close()

def populate_table(db: str, table_name: str, df: pd.DataFrame, target_col: str, db_connection: psycopg2.extensions.connection) -> None:
    db_connection = psycopg2.connect(
        host=os.environ[f"{db}-hostname"],
        user=os.environ[f"{db}-user"],
        password=os.environ[f"{db}-password"],
        dbname=os.environ[f"{db}-database"],
    )

    cur = db_connection.cursor()
    cur.execute(f"SELECT * FROM {table_name} LIMIT 0")
    col_names = [desc[0] for desc in cur.description]

    missing_columns = set(col_names).difference(df.columns)
    assert not missing_columns, f"The following columns are missing in your CSV file: {','.join(missing_columns)}"
    df = df[col_names]

    df = df.drop_duplicates(subset=[target_col, 'ts'])    
    df = df.where(pd.notnull(df), None)

    # Get the list of columns from the DataFrame
    columns = df.columns.tolist()
    
    # Generate the SET clause dynamically
    set_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in (target_col, 'ts')])

    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, sep=",")
    buffer.seek(0)

    # try:
    #     cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", buffer)
    # except psycopg2.errors.UniqueViolation:

    # Reset buffer position to start for re-reading
    buffer.seek(0)

    # Roll back the transaction to handle the error
    db_connection.rollback()

    # Reopen cursor
    cur = db_connection.cursor()

    # Temporarily create a staging table
    cur.execute(f"CREATE TEMP TABLE temp_{table_name} (LIKE {table_name} INCLUDING ALL);")
    cur.copy_expert(f"COPY temp_{table_name} FROM STDIN WITH CSV", buffer)

    # Perform upsert from staging table to the main table
    cur.execute(f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        SELECT {', '.join(columns)}
        FROM temp_{table_name}
        ON CONFLICT ({target_col}, ts) DO UPDATE
        SET {set_clause};
    """)

    # Drop the temporary staging table
    cur.execute(f"DROP TABLE temp_{table_name};")

    db_connection.commit()
    cur.close()
