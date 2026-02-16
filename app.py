import streamlit as st
import pandas as pd
import json

from connect_db import get_connection

st.set_page_config(page_title="CP Survey Dashboard", layout="wide")

conn = get_connection()

# ---------------------------
# LOAD DATA
# ---------------------------

@st.cache_data(ttl=60)
def load_data():
    intro = pd.read_sql("SELECT * FROM introductory", conn)
    ht = pd.read_sql("SELECT * FROM home_training", conn)
    it = pd.read_sql("SELECT * FROM intensive_therapies", conn)
    md = pd.read_sql("SELECT * FROM motorical_development", conn)
    return intro, ht, it, md


intro, ht, it, md = load_data()

# Merge GMFCS into motor table
md = md.merge(
    intro[["id", "gmfcs_lvl"]],
    left_on="introductory_id",
    right_on="id",
    how="left"
)

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------

st.sidebar.header("Filters")

gmfcs_options = sorted(intro["gmfcs_lvl"].dropna().unique())
selected_gmfcs = st.sidebar.multiselect(
    "GMFCS Level",
    options=gmfcs_options,
    default=gmfcs_options
)

child_options = sorted(intro["id"].unique())
selected_children = st.sidebar.multiselect(
    "Select Child",
    options=child_options,
    default=child_options
)

# Apply filters
md_filtered = md[
    (md["gmfcs_lvl"].isin(selected_gmfcs)) &
    (md["introductory_id"].isin(selected_children))
]

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def milestone_count(df, column):
    def count_milestones(x):
        if isinstance(x, dict):
            return len(x.get("milestones", []))
        return 0

    df = df.copy()
    df["milestone_count"] = df[column].apply(count_milestones)
    return df


def top_milestones_by_age(df, column, age):
    subset = df[df["age"] == age]

    milestones = []
    for item in subset[column]:
        if isinstance(item, dict):
            milestones.extend(item.get("milestones", []))

    if len(milestones) == 0:
        return pd.DataFrame()

    counts = pd.Series(milestones).value_counts().reset_index()
    counts.columns = ["Milestone", "Count"]
    return counts


# ---------------------------
# TABS
# ---------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Motor Development", "Training", "Raw Data"]
)

# =====================================================
# OVERVIEW
# =====================================================

with tab1:

    st.title("Survey Overview")

    total = len(intro)
    completed = intro["completed"].sum() if "completed" in intro else 0

    col1, col2 = st.columns(2)
    col1.metric("Total participants", total)
    col2.metric("Completed surveys", completed)

    st.subheader("Responses over time")

    entries = intro.copy()
    entries["date"] = pd.to_datetime(entries["created_at"]).dt.date
    entries = entries.groupby("date").size().reset_index(name="count")
    entries["cumulative"] = entries["count"].cumsum()

    st.line_chart(entries.set_index("date")["cumulative"])


# =====================================================
# MOTOR DEVELOPMENT
# =====================================================

with tab2:

    st.header("Motor Development")

    if md_filtered.empty:
        st.warning("No data available for selected filters.")
    else:

        # ---- GROSS ----
        st.subheader("Gross Motor Milestones")

        gross_df = milestone_count(md_filtered, "gross_motor_development")

        gross_plot = (
            gross_df
            .groupby(["age", "introductory_id"])["milestone_count"]
            .mean()
            .reset_index()
        )

        pivot_gross = gross_plot.pivot(
            index="age",
            columns="introductory_id",
            values="milestone_count"
        )

        st.line_chart(pivot_gross)

        # ---- FINE ----
        st.subheader("Fine Motor Milestones")

        fine_df = milestone_count(md_filtered, "fine_motor_development")

        fine_plot = (
            fine_df
            .groupby(["age", "introductory_id"])["milestone_count"]
            .mean()
            .reset_index()
        )

        pivot_fine = fine_plot.pivot(
            index="age",
            columns="introductory_id",
            values="milestone_count"
        )

        st.line_chart(pivot_fine)

        # ---- TOP MILESTONES ----
        st.subheader("Top Milestones by Age")

        available_ages = sorted(md_filtered["age"].dropna().unique())
        selected_age = st.selectbox("Select Age", available_ages)

        top_gross = top_milestones_by_age(md_filtered, "gross_motor_development", selected_age)

        if not top_gross.empty:
            st.bar_chart(top_gross.set_index("Milestone"))
        else:
            st.info("No milestones recorded for this age.")


# =====================================================
# TRAINING
# =====================================================

with tab3:

    st.header("Training Overview")

    ht_filtered = ht[ht["introductory_id"].isin(selected_children)]

    if ht_filtered.empty:
        st.warning("No training data available.")
    else:
        st.subheader("Training Entries Over Time")

        training_counts = (
            ht_filtered
            .groupby("introductory_id")
            .size()
            .reset_index(name="entries")
        )

        st.bar_chart(training_counts.set_index("introductory_id"))


# =====================================================
# RAW DATA
# =====================================================

with tab4:

    st.subheader("Introductory")
    st.dataframe(intro)

    st.subheader("Motor Development")
    st.dataframe(md_filtered)

    st.subheader("Home Training")
    st.dataframe(ht)

    st.subheader("Intensive Therapies")
    st.dataframe(it)