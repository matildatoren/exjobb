import streamlit as st
import pandas as pd

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

# Apply filters (consistent across tables)
intro_filtered = intro[
    (intro["gmfcs_lvl"].isin(selected_gmfcs)) &
    (intro["id"].isin(selected_children))
]

valid_ids = intro_filtered["id"].unique()

md_filtered = md[
    (md["gmfcs_lvl"].isin(selected_gmfcs)) &
    (md["introductory_id"].isin(valid_ids))
]

ht_filtered = ht[
    ht["introductory_id"].isin(valid_ids)
]

it_filtered = it[
    it["introductory_id"].isin(valid_ids)
]

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def _is_filled(x) -> bool:
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass

    if isinstance(x, str):
        return x.strip() != ""

    if isinstance(x, dict):
        if "milestones" in x:
            ms = x.get("milestones") or []
            return len(ms) > 0

        if "details" in x or "selected" in x or "other" in x:
            details = x.get("details") or {}
            selected = x.get("selected") or []
            other = (x.get("other") or "").strip()
            return (isinstance(details, dict) and len(details) > 0) or (len(selected) > 0) or (other != "")

        return len(x) > 0

    if isinstance(x, list):
        return len(x) > 0

    return True


def section_completion_per_id_age(df: pd.DataFrame, id_col: str, age_col: str, content_cols: list[str], prefix: str):
    cols = [c for c in content_cols if c in df.columns]
    if len(cols) == 0:
        return pd.DataFrame(columns=[id_col, age_col, f"{prefix}_filled_fields", f"{prefix}_total_fields"])

    tmp = df[[id_col, age_col] + cols].copy()
    filled = tmp[cols].applymap(_is_filled).sum(axis=1)
    tmp[f"{prefix}_filled_fields"] = filled
    tmp[f"{prefix}_total_fields"] = len(cols)

    out = (
        tmp.groupby([id_col, age_col], as_index=False)[[f"{prefix}_filled_fields", f"{prefix}_total_fields"]]
           .sum()
    )
    return out


def intensive_completion_per_id_age(it_df: pd.DataFrame) -> pd.DataFrame:
    """
    Conditional logic:
      - participate_therapies_neurohabilitation is ALWAYS expected (if column exists)
      - medical_treatments is ALWAYS expected (if column exists)
      - neurohabilitation_centers + methods_applied_during_intense_training count ONLY if participate == Yes
    """
    participate_col = "participate_therapies_neurohabilitation"
    centers_col = "neurohabilitation_centers"
    methods_col = "methods_applied_during_intense_training"
    medical_col = "medical_treatments"  

    needed = [c for c in [participate_col, centers_col, methods_col, medical_col] if c in it_df.columns]
    tmp = it_df[["introductory_id", "age"] + needed].copy()

    def is_yes(x) -> bool:
        if x is None:
            return False
        if isinstance(x, str):
            return x.strip().lower() in {"yes", "ja", "true", "1"}
        return False

    filled_list = []
    total_list = []

    for _, row in tmp.iterrows():
        participate_val = row.get(participate_col, None)

        base_total = 0
        base_filled = 0

        if participate_col in tmp.columns:
            base_total += 1
            base_filled += 1 if _is_filled(participate_val) else 0

        if medical_col in tmp.columns:
            base_total += 1
            base_filled += 1 if _is_filled(row.get(medical_col, None)) else 0

        cond_total = 0
        cond_filled = 0

        if is_yes(participate_val):
            if centers_col in tmp.columns:
                cond_total += 1
                cond_filled += 1 if _is_filled(row.get(centers_col, None)) else 0
            if methods_col in tmp.columns:
                cond_total += 1
                cond_filled += 1 if _is_filled(row.get(methods_col, None)) else 0

        filled_list.append(base_filled + cond_filled)
        total_list.append(base_total + cond_total)

    tmp["it_filled_fields"] = filled_list
    tmp["it_total_fields"] = total_list

    out = (
        tmp.groupby(["introductory_id", "age"], as_index=False)[["it_filled_fields", "it_total_fields"]]
           .sum()
           .rename(columns={"introductory_id": "id"})
    )
    return out


def compute_progress_percent(intro_df: pd.DataFrame, ht_df: pd.DataFrame, it_df: pd.DataFrame, md_df: pd.DataFrame) -> pd.DataFrame:
    """
    Progress over the 3 sections (HT/IT/MD) across ages 1..max_age per child.
    - Intro not included.
    - Each section-year contributes fractionally based on how many relevant fields are filled.
    - IT has conditional fields that only count if participate == Yes.
    """
    ht_cols = ["training_methods_therapies", "devices", "other_training_methods_therapies"]
    md_cols = ["gross_motor_development", "fine_motor_development", "motorical_impairments_lower", "motorical_impairments_upper"]

    def max_age_by(df, id_col="introductory_id"):
        if df.empty or "age" not in df.columns or id_col not in df.columns:
            return pd.Series(dtype=float)
        return df.groupby(id_col)["age"].max()

    max_ht = max_age_by(ht_df)
    max_it = max_age_by(it_df)
    max_md = max_age_by(md_df)

    max_age = pd.concat([max_ht, max_it, max_md], axis=1).max(axis=1)
    max_age = max_age.reindex(intro_df["id"]).fillna(0).astype(int)

    # Build age grid per child: ages 1..max_age
    grid_rows = []
    for child_id, m in zip(intro_df["id"].tolist(), max_age.tolist()):
        for age in range(1, m + 1):
            grid_rows.append((child_id, age))
    grid = pd.DataFrame(grid_rows, columns=["id", "age"])

    if grid.empty:
        return intro_df[["id"]].assign(progress_pct=0.0, n_years=0)

    ht_comp = section_completion_per_id_age(ht_df, "introductory_id", "age", ht_cols, "ht").rename(columns={"introductory_id": "id"})
    md_comp = section_completion_per_id_age(md_df, "introductory_id", "age", md_cols, "md").rename(columns={"introductory_id": "id"})
    it_comp = intensive_completion_per_id_age(it_df)

    ht_total_fields = ht_comp["ht_total_fields"].max() if not ht_comp.empty else len([c for c in ht_cols if c in ht_df.columns])
    md_total_fields = md_comp["md_total_fields"].max() if not md_comp.empty else len([c for c in md_cols if c in md_df.columns])

    merged = (
        grid.merge(ht_comp, on=["id", "age"], how="left")
            .merge(it_comp, on=["id", "age"], how="left")
            .merge(md_comp, on=["id", "age"], how="left")
    )

    for col in ["ht_filled_fields", "it_filled_fields", "md_filled_fields"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged["ht_total_fields"] = ht_total_fields
    merged["md_total_fields"] = md_total_fields

    # If IT row missing for a given age, treat as 0 filled out of base_total (participate + medical), if those columns exist
    it_base_total = 0
    if "participate_therapies_neurohabilitation" in it_df.columns:
        it_base_total += 1
    if "medical_treatments" in it_df.columns:
        it_base_total += 1

    merged["it_total_fields"] = merged["it_total_fields"].fillna(it_base_total)
    merged["it_filled_fields"] = merged["it_filled_fields"].fillna(0)

    merged["total_fields_per_age"] = merged["ht_total_fields"] + merged["it_total_fields"] + merged["md_total_fields"]
    merged["filled_fields_per_age"] = merged["ht_filled_fields"] + merged["it_filled_fields"] + merged["md_filled_fields"]

    per_child = merged.groupby("id", as_index=False).agg(
        filled=("filled_fields_per_age", "sum"),
        total=("total_fields_per_age", "sum"),
        n_years=("age", "nunique"),
    )

    per_child["progress_pct"] = per_child.apply(
        lambda r: 0.0 if r["total"] == 0 else (100.0 * r["filled"] / r["total"]),
        axis=1
    )

    return per_child[["id", "progress_pct", "n_years"]]


# ---------------------------
# TABS
# ---------------------------

tab1, tab2, tab3 = st.tabs(
    ["Overview", "Raw Data", "Completeness"]
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

    st.subheader("Participants (names)")

    cols_to_show = []
    for c in ["nick_name", "gmfcs_lvl", "completed", "created_at", "id"]:
        if c in intro.columns:
            cols_to_show.append(c)

    progress = compute_progress_percent(intro, ht, it, md)

    overview_people = intro[cols_to_show].merge(
        progress[["id", "progress_pct", "n_years"]],
        on="id",
        how="left"
    )
    overview_people["progress_pct"] = overview_people["progress_pct"].fillna(0).round(1)
    overview_people["n_years"] = overview_people["n_years"].fillna(0).astype(int)

    if "created_at" in overview_people.columns:
        overview_people["created_at"] = pd.to_datetime(overview_people["created_at"])
        overview_people = overview_people.sort_values("created_at", ascending=False)

    rename_map = {
        "nick_name": "Name",
        "gmfcs_lvl": "GMFCS",
        "completed": "Completed",
        "created_at": "Created",
        "id": "Intro ID",
        "progress_pct": "Progress (%)",
        "n_years": "Years observed",
    }
    overview_people = overview_people.rename(columns={k: v for k, v in rename_map.items() if k in overview_people.columns})

    st.dataframe(overview_people, use_container_width=True)



# =====================================================
# RAW DATA
# =====================================================

with tab2:
    st.subheader("Introductory")
    st.dataframe(intro_filtered)

    st.subheader("Home Training")
    st.dataframe(ht_filtered)

    st.subheader("Intensive Therapies")
    st.dataframe(it_filtered)

    st.subheader("Motor Development")
    st.dataframe(md_filtered)

# =====================================================
# COMPLETENESS (existing simple view kept)
# =====================================================

with tab3:
    st.header("Survey Completeness")

    ht_counts = ht.groupby("introductory_id").size().reset_index(name="ht_entries").rename(columns={"introductory_id": "id"})
    md_counts = md.groupby("introductory_id").size().reset_index(name="md_entries").rename(columns={"introductory_id": "id"})
    it_counts = it.groupby("introductory_id").size().reset_index(name="it_entries").rename(columns={"introductory_id": "id"})

    md_nulls = md.groupby("introductory_id").agg(
        gross_nulls=("gross_motor_development", lambda x: x.isna().sum()),
        fine_nulls=("fine_motor_development", lambda x: x.isna().sum()),
    ).reset_index().rename(columns={"introductory_id": "id"})

    completeness = (
        intro[["id", "nick_name", "gmfcs_lvl", "completed"]]
        .merge(ht_counts, on="id", how="left")
        .merge(md_counts, on="id", how="left")
        .merge(it_counts, on="id", how="left")
        .merge(md_nulls, on="id", how="left")
        .fillna(0)
    )

    completeness["completeness_score"] = (
        completeness[["ht_entries", "it_entries", "md_entries"]].mean(axis=1)
    ).round(1)

    def highlight_suspicious(row):
        ht_e = row["ht_entries"]
        it_e = row["it_entries"]
        md_e = row["md_entries"]

        if ht_e == md_e == it_e and row["completed"] and ht_e >= 1:
            return ["background-color: #e0ffe0"] * len(row)
        elif ht_e == md_e == it_e and ht_e >= 3:
            return ["background-color: #e0ffe0"] * len(row)
        return [""] * len(row)

    display_cols = [
        "id", "nick_name", "gmfcs_lvl", "completed",
        "ht_entries", "it_entries", "md_entries",
        "gross_nulls", "fine_nulls", "completeness_score"
    ]
    display_cols = [c for c in display_cols if c in completeness.columns]

    st.dataframe(
        completeness[display_cols].style.apply(highlight_suspicious, axis=1),
        use_container_width=True
    )

    st.subheader("Participation Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        count_ht = (completeness["ht_entries"] > 0).sum()
        st.metric("Filled in Home Training", f"{count_ht} / {len(completeness)}")

    with col2:
        count_it = (completeness["it_entries"] > 0).sum()
        st.metric("Filled in Intensive Therapies", f"{count_it} / {len(completeness)}")

    with col3:
        count_md = (completeness["md_entries"] > 0).sum()
        st.metric("Filled in Motor Development", f"{count_md} / {len(completeness)}")