"""
Glassdoor Data Science Jobs Dashboard
=====================================

Interactive Streamlit dashboard built from:
    featured_jobs.csv

Includes:
    • Key metrics
    • Salary distribution
    • Salary by job role
    • Job-role distribution
    • Experience distribution
    • Remote vs in-person
    • Top technical skills
    • Top hiring states
    • Interactive filtered data
    • CSV download
    • AI salary prediction

The salary prediction model is produced by:
    train_model.py

Required model files:
    salary_prediction_model.pkl
    salary_prediction_meta.json

Run:
    python -m streamlit run app.py
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Data Science Jobs Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME / COLORS
# ============================================================

PURPLE = {
    "dark": "#4B0082",
    "mauve": "#9370DB",
    "light": "#C8A2C8",
    "very_light": "#F5F0FA",
}

# Canonical, human-logical ordering for Experience_Group. This exists
# because the raw values are plain strings ("0-2 Years", "10+ Years",
# "3-5 Years", "6-10 Years"), and sorting them alphabetically (as
# Python's sorted() does when the model metadata was built) produces
# 0-2, 10+, 3-5, 6-10 instead of the intended youngest -> most senior
# progression. Anything that lists or orders experience buckets --
# charts, dropdowns -- should sort against this list, not alphabetically.
EXPERIENCE_ORDER = [
    "0-2 Years",
    "3-5 Years",
    "6-10 Years",
    "10+ Years",
    "Unknown",
]


def sort_by_experience_order(values):
    """Return `values` sorted into EXPERIENCE_ORDER; anything not in
    the canonical list (shouldn't normally happen) is appended at the
    end, alphabetically, rather than silently dropped."""
    known = [v for v in EXPERIENCE_ORDER if v in values]
    unknown = sorted(v for v in values if v not in EXPERIENCE_ORDER)
    return known + unknown

# US state code -> full name, used to display readable state/country
# labels in the Geography tab and the salary predictor instead of raw
# two-letter USPS codes. The underlying values passed to the model /
# used for filtering stay as the original codes.
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico",
}


def state_full_name(code):
    """Return the full state name for a USPS code, or the code itself if unknown."""
    code = str(code).strip().upper()
    return US_STATES.get(code, code)


sns.set_theme(style="whitegrid")

plt.rcParams["axes.titlecolor"] = PURPLE["dark"]
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#444444"
plt.rcParams["ytick.color"] = "#444444"


# ============================================================
# CUSTOM CSS
# ============================================================
# NOTE: every line below starts at column 0. Markdown treats any
# line indented 4+ spaces as a literal code block, which was
# silently breaking the CSS/HTML rendering further down in the
# original file (visible as raw "<div ...>" text on screen).

st.markdown(
f"""
<style>
.main {{
padding-top: 1rem;
}}
.hero {{
padding: 2rem 2.2rem;
border-radius: 18px;
margin-bottom: 1.5rem;
background: linear-gradient(135deg, #4B0082 0%, #6A3D9A 50%, #9370DB 100%);
color: white;
}}
.hero-title {{
font-size: 2.7rem;
font-weight: 800;
margin-bottom: 0.3rem;
}}
.hero-subtitle {{
font-size: 1.05rem;
opacity: 0.92;
margin-bottom: 0;
}}
.prediction-card {{
padding: 2rem;
border-radius: 18px;
background: {PURPLE["very_light"]};
border: 1px solid {PURPLE["light"]};
text-align: center;
margin-top: 1rem;
margin-bottom: 1.5rem;
}}
.prediction-label {{
font-size: 1rem;
color: #666666;
}}
.prediction-value {{
font-size: 3.2rem;
font-weight: 800;
color: {PURPLE["dark"]};
margin: 0.2rem 0;
}}
.prediction-range {{
font-size: 0.95rem;
color: #666666;
}}
.section-title {{
font-size: 1.4rem;
font-weight: 700;
color: {PURPLE["dark"]};
margin-top: 0.8rem;
margin-bottom: 0.8rem;
}}
.skill-tag {{
display: inline-block;
padding: 0.35rem 0.7rem;
margin: 0.2rem;
border-radius: 999px;
background: {PURPLE["light"]};
color: #ffffff;
font-size: 0.85rem;
font-weight: 600;
}}
.footer {{
text-align: center;
color: #777777;
font-size: 0.85rem;
padding: 1.5rem;
}}
</style>
""",
unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "featured_jobs.csv"
MODEL_PATH = BASE_DIR / "salary_prediction_model.pkl"

META_PATH = BASE_DIR / "salary_prediction_meta.json"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data(path):
    """Load engineered dashboard dataset."""
    return pd.read_csv(path)


@st.cache_resource
def load_model(model_path, meta_path):
    """
    Load trained sklearn pipeline and metadata.

    Returns:
        (model, metadata)
    """

    if not model_path.exists():
        return None, None

    if not meta_path.exists():
        return None, None

    model = joblib.load(model_path)

    with open(
        meta_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return model, metadata


# ============================================================
# LOAD DATASET
# ============================================================

if not DATA_PATH.exists():

    st.error(
        "❌ Dataset not found."
    )

    st.code(
        str(DATA_PATH)
    )

    st.info(
        "Make sure `featured_jobs1.csv` exists inside the "
        "`data` folder."
    )

    st.stop()


try:

    df = load_data(DATA_PATH)

except Exception as error:

    st.error(
        "❌ Could not load the dataset."
    )

    st.exception(error)

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, model_meta = load_model(
        MODEL_PATH,
        META_PATH,
    )

except Exception as error:

    model = None
    model_meta = None

    st.sidebar.warning(
        "⚠️ Model could not be loaded."
    )


# ============================================================
# HEADER
# ============================================================
# Flattened to column 0 with NO blank line between the title and
# subtitle divs. A blank line in the middle of this block causes
# Streamlit/CommonMark to close the first HTML block there, and the
# subtitle div (previously indented 4 spaces) then got parsed as an
# indented code block instead of HTML — that's the bug you saw.

st.markdown(
"""
<div class="hero">
<div class="hero-title">📊 Data Science Jobs Dashboard</div>
<div class="hero-subtitle">
Explore salaries, job roles, experience,
technical skills, locations, and hiring trends.
<br>
Use the AI salary predictor to estimate compensation
for a hypothetical job posting.
</div>
</div>
""",
unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 Dashboard Filters")

st.sidebar.caption(
    "Use the filters below to explore the job market."
)


# ------------------------------------------------------------
# JOB ROLE FILTER
# ------------------------------------------------------------

if "Job_Simplified" in df.columns:

    job_roles = sorted(
        df["Job_Simplified"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_roles = st.sidebar.multiselect(
        "Job Role",
        options=job_roles,
        default=job_roles,
    )

else:

    selected_roles = []


# ------------------------------------------------------------
# STATE FILTER
# ------------------------------------------------------------

if "Job_State" in df.columns:

    states = sorted(
        df["Job_State"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_states = st.sidebar.multiselect(
        "State",
        options=states,
        default=states,
        format_func=state_full_name,
    )

else:

    selected_states = []


# ------------------------------------------------------------
# REMOTE FILTER
# ------------------------------------------------------------

remote_option = st.sidebar.radio(
    "Work Type",
    options=[
        "All",
        "Remote Only",
        "In-Person Only",
    ],
    index=0,
)


# ------------------------------------------------------------
# SALARY FILTER
# ------------------------------------------------------------

salary_min = int(
    df["Avg_Salary"].min()
)

salary_max = int(
    df["Avg_Salary"].max()
)

salary_range = st.sidebar.slider(
    "Average Salary Range (K USD)",
    min_value=salary_min,
    max_value=salary_max,
    value=(
        salary_min,
        salary_max,
    ),
)


# ------------------------------------------------------------
# SIDEBAR INFO
# ------------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("📁 Dataset")

st.sidebar.write(
    f"Rows: **{len(df):,}**"
)

st.sidebar.write(
    f"Columns: **{len(df.columns):,}**"
)

st.sidebar.caption(
    "Source: Glassdoor Data Science Jobs"
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df.copy()


if selected_roles:

    filtered = filtered[
        filtered["Job_Simplified"]
        .astype(str)
        .isin(selected_roles)
    ]

else:

    filtered = filtered.iloc[0:0]


if selected_states:

    filtered = filtered[
        filtered["Job_State"]
        .astype(str)
        .isin(selected_states)
    ]

else:

    filtered = filtered.iloc[0:0]


filtered = filtered[
    filtered["Avg_Salary"].between(
        salary_range[0],
        salary_range[1],
    )
]


if remote_option == "Remote Only":

    filtered = filtered[
        filtered["Remote_Job"] == 1
    ]

elif remote_option == "In-Person Only":

    filtered = filtered[
        filtered["Remote_Job"] == 0
    ]


# ============================================================
# EMPTY FILTER RESULT
# ============================================================

if filtered.empty:

    st.warning(
        "⚠️ No postings match the current filters."
    )

    st.info(
        "Try selecting more job roles, states, "
        "or widening the salary range."
    )

    st.stop()


# ============================================================
# KEY METRICS
# ============================================================

skill_cols_all = [
    column
    for column in filtered.columns
    if column.strip().lower().endswith("_yn")
]


if skill_cols_all:

    skill_totals = (
        filtered[skill_cols_all]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if len(skill_totals) > 0:

        top_skill = (
            skill_totals.index[0]
        )

        top_skill = top_skill[: -len("_yn")] if top_skill.lower().endswith("_yn") else top_skill

    else:

        top_skill = "N/A"

else:

    top_skill = "N/A"


st.markdown(
    '<div class="section-title">📌 Market Overview</div>',
    unsafe_allow_html=True,
)


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.metric(
        "Job Postings",
        f"{len(filtered):,}",
    )


with k2:

    st.metric(
        "Avg. Salary",
        f"${filtered['Avg_Salary'].mean():.0f}K",
    )


with k3:

    st.metric(
        "Avg. Rating",
        f"{filtered['Rating'].mean():.1f} ⭐",
    )


with k4:

    st.metric(
        "Remote Share",
        f"{filtered['Remote_Job'].mean() * 100:.0f}%",
    )


with k5:

    st.metric(
        "Top Skill",
        top_skill,
    )


st.markdown("---")


# ============================================================
# TABS
# ============================================================

(
    tab_salary,
    tab_roles,
    tab_skills,
    tab_geo,
    tab_predict,
    tab_data,
) = st.tabs(
    [
        "💰 Salary",
        "👩‍💻 Roles & Experience",
        "🛠️ Skills",
        "🗺️ Geography",
        "🔮 Predict Salary",
        "📄 Raw Data",
    ]
)


# ============================================================
# SALARY TAB
# ============================================================

with tab_salary:

    st.subheader(
        "💰 Salary Analysis"
    )

    # ----------------------------------------------------------
    # AUTO-GENERATED NARRATIVE
    # ----------------------------------------------------------
    # Recomputed from `filtered`, so it updates with the sidebar
    # filters instead of describing the whole unfiltered dataset.

    role_salary_avg = (
        filtered
        .groupby("Job_Simplified")["Avg_Salary"]
        .mean()
        .sort_values(ascending=False)
    )

    if len(role_salary_avg) > 0:

        highest_role = role_salary_avg.index[0]
        highest_role_salary = role_salary_avg.iloc[0]

        lowest_role = role_salary_avg.index[-1]
        lowest_role_salary = role_salary_avg.iloc[-1]

        salary_spread = (
            filtered["Avg_Salary"].max()
            - filtered["Avg_Salary"].min()
        )

        exp_salary_avg = (
            filtered
            .groupby("Experience_Group")["Avg_Salary"]
            .mean()
            .sort_values(ascending=False)
        )

        top_exp_group = (
            exp_salary_avg.index[0]
            if len(exp_salary_avg) > 0
            else "N/A"
        )

        st.info(
            f"💡 **{highest_role}** roles pay the most in the current "
            f"selection, averaging **\\${highest_role_salary:.0f}K** — "
            f"compared to **\\${lowest_role_salary:.0f}K** for "
            f"**{lowest_role}**, the lowest-paying role shown. "
            f"Postings requiring **{top_exp_group}** of experience "
            f"command the highest average pay. Salaries in this view "
            f"range across **\\${salary_spread:.0f}K**."
        )

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # SALARY DISTRIBUTION
    # --------------------------------------------------------

    with col1:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.histplot(
            filtered["Avg_Salary"],
            bins=25,
            kde=True,
            color=PURPLE["light"],
            ax=ax,
        )

        # Sample size in the title so the shape of the histogram is
        # never read without knowing how many postings it's built
        # from (a 25-bin histogram over a handful of filtered rows
        # looks very different from one over the full dataset).
        ax.set_title(
            f"Salary Distribution (n = {len(filtered)})"
        )

        ax.set_xlabel(
            "Average Salary (K USD)"
        )

        ax.set_ylabel(
            "Number of Postings"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


    # --------------------------------------------------------
    # SALARY BY ROLE
    # --------------------------------------------------------

    with col2:

        order = (
            filtered
            .groupby("Job_Simplified")[
                "Avg_Salary"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .index
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.boxplot(
            data=filtered,
            x="Job_Simplified",
            y="Avg_Salary",
            order=order,
            color=PURPLE["light"],
            ax=ax,
        )

        ax.set_title(
            "Salary by Job Role"
        )

        ax.set_xlabel(
            "Job Role"
        )

        ax.set_ylabel(
            "Average Salary (K USD)"
        )

        # Sample size per role, right on the x-tick label. A box built
        # from 5 postings and a box built from 150 postings should not
        # look equally trustworthy, so the count goes on the axis
        # itself instead of requiring a trip to another tab.
        role_counts = filtered["Job_Simplified"].value_counts()

        ax.set_xticks(
            range(len(order))
        )

        ax.set_xticklabels(
            [
                f"{role}\n(n={role_counts.get(role, 0)})"
                for role in order
            ]
        )

        plt.xticks(
            rotation=45,
            ha="right",
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


# ============================================================
# ROLES & EXPERIENCE
# ============================================================

with tab_roles:

    st.subheader(
        "👩‍💻 Roles & Experience"
    )

    # ----------------------------------------------------------
    # AUTO-GENERATED NARRATIVE
    # ----------------------------------------------------------

    role_counts = filtered["Job_Simplified"].value_counts()

    if len(role_counts) > 0:

        top_role_name = role_counts.index[0]
        top_role_count = role_counts.iloc[0]
        top_role_pct = (top_role_count / len(filtered)) * 100

        exp_mode_counts = filtered["Experience_Group"].value_counts()

        top_exp_name = (
            exp_mode_counts.index[0]
            if len(exp_mode_counts) > 0
            else "N/A"
        )

        top_exp_pct = (
            (exp_mode_counts.iloc[0] / len(filtered)) * 100
            if len(exp_mode_counts) > 0
            else 0
        )

        remote_pct = filtered["Remote_Job"].mean() * 100

        st.info(
            f"💡 **{top_role_name}** is the most common role in the "
            f"current selection, making up **{top_role_pct:.0f}%** of "
            f"postings. Most listings ask for **{top_exp_name}** of "
            f"experience (**{top_exp_pct:.0f}%** of postings), and "
            f"only **{remote_pct:.0f}%** are remote."
        )

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # JOB ROLE DISTRIBUTION
    # --------------------------------------------------------

    with col1:

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        sns.countplot(
            data=filtered,
            y="Job_Simplified",
            order=(
                filtered[
                    "Job_Simplified"
                ]
                .value_counts()
                .index
            ),
            color=PURPLE["light"],
            ax=ax,
        )

        ax.set_title(
            "Job Role Distribution"
        )

        ax.set_xlabel(
            "Number of Postings"
        )

        ax.set_ylabel(
            "Job Role"
        )

        # Exact count at the end of each bar, same treatment as the
        # Skills, Experience, and Remote charts already get -- this
        # was the one count-style chart in the dashboard without its
        # own on-bar numbers.
        ax.bar_label(
            ax.containers[0],
            padding=3,
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    with col2:

        exp_order = EXPERIENCE_ORDER

        exp_counts = (
            filtered[
                "Experience_Group"
            ]
            .value_counts()
            .reindex(exp_order)
            .fillna(0)
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        bar = sns.barplot(
            x=exp_counts.index,
            y=exp_counts.values,
            color=PURPLE["light"],
            ax=ax,
        )

        for i, value in enumerate(
            exp_counts.values
        ):

            bar.text(
                i,
                value,
                f"{int(value)}",
                ha="center",
                va="bottom",
            )

        ax.set_title(
            "Experience Required"
        )

        ax.set_xlabel(
            "Experience"
        )

        ax.set_ylabel(
            "Number of Postings"
        )

        plt.xticks(
            rotation=30,
            ha="right",
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


    # --------------------------------------------------------
    # REMOTE VS IN-PERSON
    # --------------------------------------------------------

    with col3:

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        sns.countplot(
            data=filtered,
            x="Remote_Job",
            color=PURPLE["light"],
            ax=ax,
        )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_xticklabels(
            [
                "In-Person",
                "Remote",
            ]
        )

        ax.set_title(
            "Remote vs In-Person"
        )

        ax.set_xlabel(
            "Work Type"
        )

        ax.set_ylabel(
            "Number of Postings"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


# ============================================================
# SKILLS TAB
# ============================================================

with tab_skills:

    st.subheader("🛠️ Technical Skills")

    if skill_cols_all:

        skill_counts = (
            filtered[skill_cols_all]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        skill_counts.columns = ["Skill", "Count"]

        skill_counts["Skill"] = (
            skill_counts["Skill"]
            .str.replace(
                r"_yn$",
                "",
                regex=True,
                flags=re.IGNORECASE,
            )
        )

        # ------------------------------------------------------
        # AUTO-GENERATED NARRATIVE
        # ------------------------------------------------------

        if len(skill_counts) > 0:

            top_3 = skill_counts.head(3)

            top_3_text = ", ".join(
                f"**{row.Skill}** ({row.Count} postings)"
                for row in top_3.itertuples()
            )

            rarest_skill_row = skill_counts.iloc[-1]

            rarest_pct = (
                rarest_skill_row["Count"] / len(filtered)
            ) * 100

            st.info(
                f"💡 The most in-demand skills in the current "
                f"selection are {top_3_text}. At the other end, "
                f"**{rarest_skill_row['Skill']}** is the rarest, "
                f"appearing in only **{rarest_pct:.0f}%** of postings."
            )

        # ------------------------------------------------------
        # SKILL BAR CHART
        # ------------------------------------------------------

        n_skills = len(skill_counts)
        fig_height = max(6, 0.45 * n_skills)

        fig, ax = plt.subplots(figsize=(10, fig_height))

        bar = sns.barplot(
            data=skill_counts,
            x="Count",
            y="Skill",
            color=PURPLE["light"],
            ax=ax,
        )

        for i, value in enumerate(skill_counts["Count"]):
            bar.text(
                value + 0.5,
                i,
                str(int(value)),
                va="center",
            )

        ax.set_title("Most Requested Skills & Tools")
        ax.set_xlabel("Number of Postings")
        ax.set_ylabel("Skill")

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    else:

        st.info("No `*_yn` skill columns were found.")
# ============================================================
# GEOGRAPHY TAB
# ============================================================

with tab_geo:

    st.subheader(
        "🗺️ Hiring Geography"
    )

    full_state_counts = filtered["Job_State"].value_counts()

    state_counts = full_state_counts.head(15)

    state_counts.index = [
        state_full_name(code)
        for code in state_counts.index
    ]

    # ------------------------------------------------------------
    # AUTO-GENERATED NARRATIVE
    # ------------------------------------------------------------

    if len(full_state_counts) > 0:

        top_state_name = state_full_name(full_state_counts.index[0])
        top_state_pct = (
            full_state_counts.iloc[0] / len(filtered)
        ) * 100

        top_3_states_pct = (
            full_state_counts.head(3).sum() / len(filtered)
        ) * 100

        st.info(
            f"💡 **{top_state_name}** leads hiring in the current "
            f"selection with **{top_state_pct:.0f}%** of postings. "
            f"The top 3 states together account for "
            f"**{top_3_states_pct:.0f}%** of all postings shown, "
            f"showing how concentrated hiring is geographically."
        )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.barplot(
        x=state_counts.values,
        y=state_counts.index,
        color=PURPLE["mauve"],
        ax=ax,
    )

    # Exact posting count at the end of each state's bar.
    ax.bar_label(
        ax.containers[0],
        padding=3,
    )

    # Make the top-15 cutoff explicit in the title -- otherwise a
    # filtered view with fewer than 15 states on screen looks
    # identical to "this chart is always capped at 15", when it's
    # actually showing every state that matched the current filter.
    n_states_total = len(full_state_counts)
    n_states_shown = len(state_counts)

    if n_states_total > n_states_shown:
        title = (
            f"Top Hiring States "
            f"(top {n_states_shown} of {n_states_total})"
        )
    else:
        title = (
            f"Top Hiring States (all {n_states_total} shown)"
        )

    ax.set_title(title)

    ax.set_xlabel(
        "Number of Postings"
    )

    ax.set_ylabel(
        "State"
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# ============================================================
# PREDICT SALARY TAB
# ============================================================

with tab_predict:

    st.subheader(
        "🔮 AI Salary Predictor"
    )

    st.write(
        "Enter job characteristics and technical skills "
        "to estimate the average salary predicted by the "
        "trained machine-learning model."
    )


    # --------------------------------------------------------
    # MODEL NOT AVAILABLE
    # --------------------------------------------------------

    if model is None:

        st.warning(
            "⚠️ Trained salary prediction model not found."
        )

        st.markdown(
            """
            ### Generate the model first

            Run your training script:

            ```bash
            python train_model.py
            ```

            This should create:

            ```text
            salary_prediction_model.pkl
            salary_prediction_meta.json
            ```

            Both files should be located next to `app.py`.
            """
        )


    # --------------------------------------------------------
    # MODEL AVAILABLE
    # --------------------------------------------------------

    else:

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        model_name = model_meta.get(
            "model_name",
            "Unknown",
        )

        test_mae = float(
            model_meta.get(
                "test_mae",
                0,
            )
        )

        test_rmse = float(
            model_meta.get(
                "test_rmse",
                0,
            )
        )

        test_r2 = float(
            model_meta.get(
                "test_r2",
                0,
            )
        )

        n_rows = int(
            model_meta.get(
                "n_rows",
                len(df),
            )
        )

        st.markdown(
            "### 📊 Model Performance"
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Model",
                model_name,
            )

        with m2:

            st.metric(
                "Test MAE",
                f"${test_mae:.1f}K",
            )

        with m3:

            st.metric(
                "Test RMSE",
                f"${test_rmse:.1f}K",
            )

        with m4:

            st.metric(
                "R² Score",
                f"{test_r2:.3f}",
            )


        st.caption(
            f"The model was trained using {n_rows:,} "
            "records. MAE represents the average absolute "
            "prediction error on the holdout test set."
        )


        st.markdown("---")


        # ====================================================
        # INPUTS
        # ====================================================

        st.markdown(
            "### 📝 Job Information"
        )

        input_col1, input_col2 = st.columns(2)


        # ----------------------------------------------------
        # LEFT INPUTS
        # ----------------------------------------------------

        with input_col1:

            job_options = model_meta[
                "categorical_options"
            ].get(
                "Job_Simplified",
                [],
            )

            state_options = model_meta[
                "categorical_options"
            ].get(
                "Job_State",
                [],
            )

            experience_options = sort_by_experience_order(
                model_meta[
                    "categorical_options"
                ].get(
                    "Experience_Group",
                    [],
                )
            )


            role_input = st.selectbox(
                "Job Role",
                options=job_options,
            )


            state_input = st.selectbox(
                "State",
                options=state_options,
                format_func=state_full_name,
            )


            experience_input = st.selectbox(
                "Experience Required",
                options=experience_options,
            )


        # ----------------------------------------------------
        # RIGHT INPUTS
        # ----------------------------------------------------

        with input_col2:

            rating_range = model_meta[
                "numeric_ranges"
            ].get(
                "Rating",
                [1.0, 5.0],
            )

            rating_min = float(
                rating_range[0]
            )

            rating_max = float(
                rating_range[1]
            )

            default_rating = min(
                max(
                    3.5,
                    rating_min,
                ),
                rating_max,
            )

            rating_input = st.slider(
                "Company Rating",
                min_value=rating_min,
                max_value=rating_max,
                value=default_rating,
                step=0.1,
            )


            age_range = model_meta[
                "numeric_ranges"
            ].get(
                "Company_Age",
                [0.0, 100.0],
            )

            age_min = float(
                age_range[0]
            )

            age_max = float(
                age_range[1]
            )

            default_age = min(
                max(
                    15.0,
                    age_min,
                ),
                age_max,
            )


            age_input = st.number_input(
                "Company Age (years)",
                min_value=age_min,
                max_value=age_max,
                value=default_age,
                step=1.0,
            )


            remote_input = st.checkbox(
                "🏠 Remote position"
            )


        # ====================================================
        # SKILLS
        # ====================================================

        st.markdown(
            "### 🛠️ Technical Skills"
        )

        st.caption(
            "Select the technical skills relevant to this "
            "job posting."
        )

        skill_columns = model_meta.get(
            "skill_features",
            [],
        )

        skill_labels = [
            column.replace(
                "_yn",
                "",
            )
            for column in skill_columns
        ]

        selected_skills = st.multiselect(
            "Skills / Tools",
            options=skill_labels,
        )


        # ====================================================
        # INPUT PREVIEW
        # ====================================================

        with st.expander(
            "🔍 Preview model input"
        ):

            preview_row = {
                "Job_Simplified": role_input,
                "Job_State": state_input,
                "Experience_Group": experience_input,
                "Rating": rating_input,
                "Company_Age": age_input,
                "Remote_Job": int(
                    remote_input
                ),
            }

            for skill_column in skill_columns:

                skill_name = (
                    skill_column
                    .replace(
                        "_yn",
                        "",
                    )
                )

                preview_row[
                    skill_column
                ] = (
                    1
                    if skill_name
                    in selected_skills
                    else 0
                )


            preview_df = pd.DataFrame(
                [preview_row]
            )


            st.dataframe(
                preview_df,
                use_container_width=True,
                hide_index=True,
            )


        # ====================================================
        # PREDICT BUTTON
        # ====================================================

        st.markdown("")


        predict_clicked = st.button(
            "🚀 Predict Salary",
            type="primary",
            use_container_width=True,
        )


        if predict_clicked:

            try:

                # --------------------------------------------
                # BUILD INPUT ROW
                # --------------------------------------------

                row = {
                    "Job_Simplified": role_input,
                    "Job_State": state_input,
                    "Experience_Group": experience_input,
                    "Rating": rating_input,
                    "Company_Age": age_input,
                    "Remote_Job": int(
                        remote_input
                    ),
                }


                # --------------------------------------------
                # ADD ALL SKILL COLUMNS
                # --------------------------------------------

                for skill_column in skill_columns:

                    skill_name = (
                        skill_column
                        .replace(
                            "_yn",
                            "",
                        )
                    )

                    row[
                        skill_column
                    ] = (
                        1
                        if skill_name
                        in selected_skills
                        else 0
                    )


                # --------------------------------------------
                # CREATE DATAFRAME
                # --------------------------------------------

                input_df = pd.DataFrame(
                    [row]
                )


                # --------------------------------------------
                # IMPORTANT:
                # EXACT FEATURE ORDER
                # --------------------------------------------

                expected_features = model_meta.get(
                    "all_features",
                    list(
                        input_df.columns
                    ),
                )


                missing_features = [
                    feature
                    for feature
                    in expected_features
                    if feature
                    not in input_df.columns
                ]


                if missing_features:

                    raise ValueError(
                        "The following model features "
                        "are missing: "
                        + str(
                            missing_features
                        )
                    )


                input_df = input_df[
                    expected_features
                ]


                # --------------------------------------------
                # PREDICTION
                # --------------------------------------------

                prediction = float(
                    model.predict(
                        input_df
                    )[0]
                )


                # Salary cannot logically be negative.
                prediction = max(
                    prediction,
                    0,
                )


                # --------------------------------------------
                # APPROXIMATE ERROR BAND
                # --------------------------------------------

                lower = max(
                    prediction - test_mae,
                    0,
                )

                upper = (
                    prediction + test_mae
                )


                # --------------------------------------------
                # RESULT
                # --------------------------------------------

                st.markdown("---")

                st.markdown(
                    "### 🎯 Prediction Result"
                )

                # Flattened to column 0, single continuous HTML
                # block with no blank lines in the middle — this
                # is the same fix as the hero header above and is
                # what was rendering as raw "<div ...>" text
                # in the "Prediction Result" screenshot.
                st.markdown(
f"""
<div class="prediction-card">
<div class="prediction-label">Estimated Average Salary</div>
<div class="prediction-value">${prediction:,.1f}K</div>
<div class="prediction-range">
Approximate MAE-based range:
<strong>${lower:,.1f}K – ${upper:,.1f}K</strong>
</div>
</div>
""",
unsafe_allow_html=True,
                )

                # --------------------------------------------
                # AUTO-GENERATED NARRATIVE
                # --------------------------------------------
                # Compares this prediction against the average
                # salary in the currently filtered dataset, so the
                # number has context instead of standing alone.

                filtered_avg_salary = filtered["Avg_Salary"].mean()

                diff_from_avg = prediction - filtered_avg_salary

                if abs(diff_from_avg) < 1:

                    comparison_text = (
                        "in line with the average salary "
                        f"(\\${filtered_avg_salary:.0f}K) across the "
                        "postings currently shown."
                    )

                else:

                    direction = (
                        "above" if diff_from_avg > 0 else "below"
                    )

                    comparison_text = (
                        f"**\\${abs(diff_from_avg):.0f}K {direction}** "
                        "the average salary "
                        f"(\\${filtered_avg_salary:.0f}K) across the "
                        "postings currently shown."
                    )

                st.info(
                    f"💡 This estimate of **\\${prediction:,.0f}K** is "
                    f"{comparison_text}"
                )


                # --------------------------------------------
                # RESULT METRICS
                # --------------------------------------------

                r1, r2, r3 = st.columns(3)


                with r1:

                    st.metric(
                        "Estimated Salary",
                        f"${prediction:,.1f}K",
                    )


                with r2:

                    st.metric(
                        "Lower Estimate",
                        f"${lower:,.1f}K",
                    )


                with r3:

                    st.metric(
                        "Upper Estimate",
                        f"${upper:,.1f}K",
                    )


                # --------------------------------------------
                # JOB DETAILS SUMMARY
                # --------------------------------------------
                # Shows exactly what inputs the prediction above
                # was based on — role, state, experience, and the
                # other job characteristics — so the result is
                # never shown without its context.

                st.markdown(
                    "#### 📋 Based On These Job Details"
                )

                d1, d2, d3 = st.columns(3)

                with d1:

                    st.metric(
                        "Job Role",
                        role_input,
                    )

                    st.metric(
                        "State",
                        state_full_name(state_input),
                    )

                with d2:

                    st.metric(
                        "Experience",
                        experience_input,
                    )

                    st.metric(
                        "Company Rating",
                        f"{rating_input:.1f} ⭐",
                    )

                with d3:

                    st.metric(
                        "Company Age",
                        f"{age_input:.0f} yrs",
                    )

                    st.metric(
                        "Work Type",
                        "Remote" if remote_input else "In-Person",
                    )


                # --------------------------------------------
                # SELECTED SKILLS
                # --------------------------------------------

                st.markdown(
                    "#### 💻 Selected Skills"
                )


                if selected_skills:

                    skills_html = ""

                    for skill in selected_skills:
                        skills_html += f'<span class="skill-tag">{skill}</span>'


                    st.markdown(
                        skills_html,
                        unsafe_allow_html=True,
                    )

                else:

                    st.caption(
                        "No technical skills selected."
                    )


                # --------------------------------------------
                # DISCLAIMER
                # --------------------------------------------

                st.info(
                    "The range above is based on the model's "
                    "test MAE. It is an approximate indication "
                    "of typical prediction error, not a "
                    "statistical confidence interval or a "
                    "guaranteed salary range."
                )


            except Exception as error:

                st.error(
                    "❌ Prediction failed."
                )

                st.exception(error)


# ============================================================
# RAW DATA TAB
# ============================================================

with tab_data:

    st.subheader(
        "📄 Filtered Dataset"
    )

    st.write(
        f"Showing **{len(filtered):,}** postings "
        "matching the current filters."
    )


    st.dataframe(
        filtered,
        use_container_width=True,
        height=500,
    )


    # --------------------------------------------------------
    # CSV DOWNLOAD
    # --------------------------------------------------------

    csv = (
        filtered
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_jobs.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================
# Flattened to column 0 for consistency/safety, same fix as above.

st.markdown(
"""
<div class="footer">
<hr>
<strong>Data Science Jobs Dashboard</strong>
<br>
Built with Streamlit · pandas · Matplotlib · Seaborn · Scikit-learn
<br>
Data cleaning and feature engineering performed through the project notebooks.
</div>
""",
unsafe_allow_html=True,
)
