import streamlit as st
import pandas as pd

from connect_db import get_connection

st.set_page_config(page_title="Enkät-dashboard", layout="wide")

conn = get_connection()

@st.cache_data
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

users = load_user_data()
intro = load_intro_data()
ht = load_ht_data()
it = load_it_data()
md = load_md_data()

st.title("Survey Dashboard")

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