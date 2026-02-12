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

def load_latest_response():
    query = """
        SELECT MAX(created_at) AS latest_response
        FROM introductory
    """
    return pd.read_sql(query, conn)

def load_story_status_highest_age():
    query = """
        WITH highest_age AS (
            SELECT
                introductory_id,
                MAX(age) AS max_age
            FROM motorical_development
            GROUP BY introductory_id
        )

        SELECT
            ha.introductory_id,
            ha.max_age,
            md.story,
            CASE
                WHEN md.story IS NULL OR md.story = '' THEN 'Nej'
                ELSE 'Ja'
            END AS filled_story_at_highest_age
        FROM highest_age ha
        JOIN motorical_development md
            ON md.introductory_id = ha.introductory_id
            AND md.age = ha.max_age
        ORDER BY ha.introductory_id;
    """
    return pd.read_sql(query, conn)


df = load_entries_over_time()
users = load_user_data()
intro = load_intro_data()
ht = load_ht_data()
it = load_it_data()
md = load_md_data()
rt = load_rt_data()
df_latest = load_latest_response()
story_status = load_story_status_highest_age()

st.title("Survey Dashboard")

total = len(intro)
filled = (story_status["filled_story_at_highest_age"] == "Ja").sum()
not_filled = total - filled

col1, col2, col3 = st.columns(3)

col1.metric("Totalt antal svar", total)
col2.metric("Fyllt i sista sektionen", filled)
col3.metric("Ej fyllt i sista sektionen", not_filled)

progress = filled / total if total > 0 else 0
st.progress(progress)

st.caption(f"{progress:.0%} har fyllt i story på högsta age")


latest_ts = df_latest["latest_response"].iloc[0]
st.metric(
    label="Senaste inkomna svar",
    value=latest_ts.strftime("%Y-%m-%d %H:%M")
)

df["cumulative_entries"] = df["entries"].cumsum()
st.subheader("Totalt antal svar över tid (kumulativt)")
st.line_chart(
    df.set_index("date")["cumulative_entries"]
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