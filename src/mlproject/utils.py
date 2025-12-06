import os
import sys
import pandas as pd
import pymysql
from dotenv import load_dotenv

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

# Load environment variables
load_dotenv()

# Read DB credentials
host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
db = os.getenv("MYSQL_DB")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        # MySQL connection
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
            port=3306
        )

        logging.info("MySQL connection established")

        # Read data
        df = pd.read_sql_query("SELECT * FROM students", mydb)

        # ✅ PRINT df.head() (for your understanding)
        print("\n✅ Sample data from MySQL:")
        print(df.head(), "\n")

        logging.info("Data read successfully from MySQL")

        return df

    except Exception as ex:
        raise CustomException(ex, sys)
