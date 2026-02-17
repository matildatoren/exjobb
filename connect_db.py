import os
import psycopg2
import dotenv
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("DB_URL")

def get_connection():
    return psycopg2.connect(DB_DSN)
