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
    users = pd.read_sql("SELECT * FROM users", conn)
    intro = pd.read_sql("SELECT * FROM introductory", conn)
    ht = pd.read_sql("SELECT * FROM home_training", conn)
    it = pd.read_sql("SELECT * FROM intensive_therapies", conn)
    md = pd.read_sql("SELECT * FROM motorical_development", conn)

    ht_old = pd.read_sql("SELECT * FROM home_training_old", conn)
    it_old = pd.read_sql("SELECT * FROM intensive_therapies_old", conn)
    md_old = pd.read_sql("SELECT * FROM motorical_development_old", conn)

    return users, intro, ht, it, md, ht_old, it_old, md_old


users, intro, ht, it, md, ht_old, it_old, md_old = load_data()

# Merge GMFCS into motor table
md = md.merge(
    intro[["id", "gmfcs_lvl"]],
    left_on="introductory_id",
    right_on="id",
    how="left"
)

# Merge GMFCS into old motor table
md_old = md_old.merge(
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

md_old_filtered = md_old[
    (md_old["gmfcs_lvl"].isin(selected_gmfcs)) &
    (md_old["introductory_id"].isin(valid_ids))
]

ht_old_filtered = ht_old[
    ht_old["introductory_id"].isin(valid_ids)
]

it_old_filtered = it_old[
    it_old["introductory_id"].isin(valid_ids)
]

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def compute_intensive_training_status(it_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise intensive therapy participation per child based on
    participate_therapies_neurohabilitation.

    Rules:
    - Yes if the child has at least one row with Yes
    - No if the child has data but no Yes
    - No data if all values are missing/empty
    """
    col = "participate_therapies_neurohabilitation"

    if col not in it_df.columns:
        return pd.DataFrame(columns=["id", "intensive_training"])

    tmp = it_df[["introductory_id", col]].copy()

    def normalize(x):
        if pd.isna(x):
            return None
        if isinstance(x, str):
            val = x.strip().lower()
            if val in {"yes", "ja", "true", "1"}:
                return "Yes"
            if val in {"no", "nej", "false", "0"}:
                return "No"
            return x.strip()
        return str(x)

    tmp[col] = tmp[col].apply(normalize)

    def summarise(values):
        vals = [v for v in values if v not in [None, ""]]
        if not vals:
            return "No data"
        if "Yes" in vals:
            return "Yes"
        if "No" in vals:
            return "No"
        return ", ".join(sorted(set(vals)))

    out = (
        tmp.groupby("introductory_id")[col]
        .apply(summarise)
        .reset_index()
        .rename(columns={
            "introductory_id": "id",
            col: "intensive_training"
        })
    )

    return out

def _is_filled(x) -> bool:
    """
    Returnerar True om fältet anses ifyllt.

    Ändringar:
    - story, reflection räknas INTE här (de är exkluderade från col-listorna).
    - devices: räknas som ifyllt om minst ett val gjorts i 'selected'.
      Frekvensrutan ('details') är frivillig och krävs INTE för att fältet
      ska anses ifyllt.
    """
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
        # Milestone-struktur (t.ex. gross_motor_development)
        if "milestones" in x:
            ms = x.get("milestones") or []
            return len(ms) > 0

        # Val-struktur med selected/details/other (t.ex. devices, training_methods)
        if "details" in x or "selected" in x or "other" in x or "devices" in x:
            selected = x.get("selected") or x.get("devices") or []
            other = (x.get("other") or "").strip()
            details = x.get("details") or {}

            if len(selected) > 0:
                return True
            if isinstance(details, dict) and len(details) > 0:
                return True
            if other != "":
                return True
            return False

            # 'selected' räcker — 'details' (t.ex. frekvensruta för devices) är frivillig
            if len(selected) > 0:
                return True
            # details utan selected kan fortfarande vara ifyllt (t.ex. fritext-fält)
            if isinstance(details, dict) and len(details) > 0:
                return True
            if other != "":
                return True
            return False

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
    Villkorlig logik för IT-sektionen:
      - participate_therapies_neurohabilitation räknas ALLTID (om kolumnen finns)
      - medical_treatments räknas ALLTID (om kolumnen finns)
      - neurohabilitation_centers + methods_applied_during_intense_training räknas
        BARA om participate == Yes
      - story är frivillig och räknas INTE
    """
    participate_col = "participate_therapies_neurohabilitation"
    centers_col = "neurohabilitation_centers"
    methods_col = "methods_applied_during_intense_training"
    medical_col = "medical_treatments"
    # story exkluderas medvetet — frivilligt fält

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
    Progress över de 3 sektionerna (HT/IT/MD) per barn och år.

    Frivilliga fält som INTE räknas:
      - story      (HT, IT, MD)
      - reflection (HT)
      - devices-frekvens/details — devices räknas som ifyllt om selected är valt,
        oavsett om frekvensrutan fyllts i (hanteras i _is_filled)
    """
    ht_cols = [
        "training_methods_therapies",
        "devices",
        "other_training_methods_therapies",
        # "story" och "reflection" exkluderas — frivilliga
    ]
    md_cols = [
        "gross_motor_development",
        "fine_motor_development",
        "motorical_impairments_lower",
        "motorical_impairments_upper",
        # "story" exkluderas — frivillig
    ]

    def max_age_by(df, id_col="introductory_id"):
        if df.empty or "age" not in df.columns or id_col not in df.columns:
            return pd.Series(dtype=float)
        return df.groupby(id_col)["age"].max()

    max_ht = max_age_by(ht_df)
    max_it = max_age_by(it_df)
    max_md = max_age_by(md_df)

    max_age = pd.concat([max_ht, max_it, max_md], axis=1).max(axis=1)
    max_age = max_age.reindex(intro_df["id"]).fillna(0).astype(int)

    # Bygg åldersrutnät per barn: åldrar 1..max_age
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

    # Om IT-rad saknas för en given ålder, behandla som 0 ifyllt av base_total
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

    from datetime import date, timedelta

    today = date.today()
    yesterday = today - timedelta(days=1)

    intro["_date"] = pd.to_datetime(intro["created_at"]).dt.date

    total = len(intro)
    total_yesterday = (intro["_date"] <= yesterday).sum()
    delta_total = total - total_yesterday

    submitted = intro["completed"].sum() if "completed" in intro else 0
    submitted_yesterday = (
        intro[intro["_date"] <= yesterday]["completed"].sum()
        if "completed" in intro.columns else 0
    )
    delta_submitted = submitted - submitted_yesterday

    # --- Compute progress early so we can use it for the "completed" metric ---
    progress = compute_progress_percent(intro, ht, it, md)
    intensive_status = compute_intensive_training_status(it)
    intro_with_progress = intro.merge(progress[["id", "progress_pct"]], on="id", how="left")
    intro_with_progress["progress_pct"] = intro_with_progress["progress_pct"].fillna(0)
    intro_with_progress["_date"] = pd.to_datetime(intro_with_progress["created_at"]).dt.date

    # Completed = submitted (completed flag) OR 100% progress score
    fully_completed = int(
        (
            (intro_with_progress["completed"] == True) |
            (intro_with_progress["progress_pct"] >= 100)
        ).sum()
    )
    fully_completed_yesterday = int(
        (
            (intro_with_progress["_date"] <= yesterday) &
            (
                (intro_with_progress["completed"] == True) |
                (intro_with_progress["progress_pct"] >= 100)
            )
        ).sum()
    )
    delta_fully_completed = fully_completed - fully_completed_yesterday

    completed_ids = intro_with_progress.loc[
    (intro_with_progress["completed"] == True) |
    (intro_with_progress["progress_pct"] >= 100),
    "id"
]

    completed_intensive = intensive_status[intensive_status["id"].isin(completed_ids)]

    n_completed_with_data = (completed_intensive["intensive_training"] != "No data").sum()
    n_completed_yes = (completed_intensive["intensive_training"] == "Yes").sum()

    intensive_yes_pct = (
        100 * n_completed_yes / n_completed_with_data
        if n_completed_with_data > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total participants", total, delta=int(delta_total))
    col3.metric("Submitted surveys", int(submitted), delta=int(delta_submitted))
    col2.metric(
        "Completed surveys",
        fully_completed,
        delta=int(delta_fully_completed),
        help="Surveys that are submitted or filled to 100%",
    )
    col4.metric(
    "Completed with intensive therapy",
    f"{intensive_yes_pct:.1f}%",
    help=f"{n_completed_yes} of {n_completed_with_data} completed participants with IT data answered Yes",
)

    st.subheader("Responses over time")

    entries = intro.copy()
    entries["date"] = pd.to_datetime(entries["created_at"]).dt.date
    entries = entries.groupby("date").size().reset_index(name="count")
    entries["cumulative"] = entries["count"].cumsum()
    st.line_chart(entries.set_index("date")["cumulative"])

    st.subheader("Participants (names)")

    cols_to_show = []
    for c in ["nick_name", "country", "completed", "created_at", "id"]:
        if c in intro.columns:
            cols_to_show.append(c)

    # --- Beräkna senaste entry över alla tabeller ---
    def latest_created_at(df, id_col="introductory_id"):
        if "created_at" not in df.columns or id_col not in df.columns:
            return pd.DataFrame(columns=["id", "latest_entry"])
        return (
            df.groupby(id_col)["created_at"]
            .max()
            .reset_index()
            .rename(columns={id_col: "id", "created_at": "latest_entry"})
        )

    intro_latest = latest_created_at(intro, id_col="id")
    ht_latest = latest_created_at(ht)
    it_latest = latest_created_at(it)
    md_latest = latest_created_at(md)

    all_latest = (
        pd.concat([intro_latest, ht_latest, it_latest, md_latest], ignore_index=True)
        .groupby("id")["latest_entry"]
        .max()
        .reset_index()
    )
    all_latest["latest_entry"] = pd.to_datetime(all_latest["latest_entry"]).dt.date

    overview_people = intro[cols_to_show].merge(
        progress[["id", "progress_pct", "n_years"]],
        on="id",
        how="left"
    ).merge(
        all_latest,
        on="id",
        how="left"
    ).merge(
        intensive_status,
        on="id",
        how="left"
    )


    overview_people["progress_pct"] = overview_people["progress_pct"].fillna(0).round(1)
    overview_people["n_years"] = overview_people["n_years"].fillna(0).astype(int)
    overview_people["intensive_training"] = overview_people["intensive_training"].fillna("No data")

    if "created_at" in overview_people.columns:
        overview_people["created_at"] = pd.to_datetime(overview_people["created_at"])
        overview_people = overview_people.sort_values("created_at", ascending=False)

    rename_map = {
        "nick_name": "Name",
        "gmfcs_lvl": "GMFCS",
        "completed": "Submitted",
        "created_at": "Created",
        "id": "Intro ID",
        "progress_pct": "Progress (%)",
        "n_years": "Years observed",
        "latest_entry": "Latest Entry",
        "intensive_training": "Intensive therapy",
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

    st.subheader("Home Training Old")
    st.dataframe(ht_old_filtered)

    st.subheader("Intensive Therapies")
    st.dataframe(it_filtered)

    st.subheader("Intensive Therapies Old")
    st.dataframe(it_old_filtered)

    st.subheader("Motor Development")
    st.dataframe(md_filtered)

    st.subheader("Motor Development Old")
    st.dataframe(md_old_filtered)

    st.subheader("Users")
    st.dataframe(users)

    st.subheader("User ID ↔ Introductory ID Mapping")
    users_map = users[["id", "email_cryp", "created_at"]].rename(columns={"id": "user_id"}).copy()
    intro_map = intro_filtered[["id", "user_id"]].rename(columns={"id": "introductory_id"}).copy()

    users_map["user_id"] = users_map["user_id"].astype(str)
    intro_map["user_id"] = intro_map["user_id"].astype(str)

    id_map = (
        users_map
        .merge(intro_map, on="user_id", how="left")
        [["user_id", "email_cryp", "created_at", "introductory_id"]]
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    st.dataframe(id_map, use_container_width=True)

# =====================================================
# COMPLETENESS
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