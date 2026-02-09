import streamlit as st
import pandas as pd

from connect_db import get_connection

st.set_page_config(page_title="Enkät-dashboard", layout="wide")

conn = get_connection()

@st.cache_data
def load_entries_over_time():
    query = """
        SELECT
            DATE(created_at) AS date,
            COUNT(*) AS entries
        FROM introductory
        GROUP BY DATE(created_at)
        ORDER BY date
    """
    return pd.read_sql(query, conn)


def load_user_data():
    query = """
        SELECT *
        FROM users
    """
    return pd.read_sql(query, conn)

def load_intro_data():
    query = """
        SELECT *
        FROM introductory
    """
    return pd.read_sql(query, conn)

def load_ht_data():
    query = """
        SELECT *
        FROM home_training
    """
    return pd.read_sql(query, conn)


def load_it_data():
    query = """
        SELECT *
        FROM intensive_therapies 
    """
    return pd.read_sql(query, conn)


def load_md_data():
    query = """
        SELECT *
        FROM motorical_development
    """
    return pd.read_sql(query, conn)

def load_rt_data():
    query = """
        SELECT *
        FROM refresh_tokens
    """
    return pd.read_sql(query, conn)

df = load_entries_over_time()
users = load_user_data()
intro = load_intro_data()
ht = load_ht_data()
it = load_it_data()
md = load_md_data()
rt = load_rt_data()


st.title("Survey Dashboard")

st.metric("Totalt antal svar", len(intro))

st.subheader("Antal entries per dag")
st.line_chart(
    df.set_index("date")["entries"]
)

st.header("All tables displayed")
st.subheader("Users")
st.dataframe(users)

st.subheader("Introductory")
st.dataframe(intro)

st.subheader("Home Training")
st.dataframe(ht)

st.subheader("Intensive Therapies")
st.dataframe(it)

st.subheader("Motorical development")
st.dataframe(md)

st.subheader("Refresh tokens")
st.dataframe(rt)