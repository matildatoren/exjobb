import os
import psycopg2
import dotenv
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("DB_URL")

def get_connection():
    return psycopg2.connect(DB_DSN)

# try:
#     conn = psycopg2.connect(DB_DSN)
# except (Exception, psycopg2.Error) as error:
#     print("Error while connecting to PostgreSQL", error)