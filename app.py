"""
SepsiSensor — Circadian-Aware Antibiotic Concentration Simulator
================================================================

EDUCATIONAL SIMULATION ONLY — NOT FOR CLINICAL USE.

This program is a teaching tool for a science project. It compares two
*mathematical scenarios* using invented numbers:

  1. A standard fixed-interval simulated dosing schedule with constant clearance.
  2. A circadian-aware model where the simulated clearance varies with time of day.

Nothing here is a medicine, a dose, a patient, or a clinical recommendation.
All units are fictional "simulation units" and the parameters were chosen to
make the maths easy to see on a chart, not to represent any real drug.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# NumPy 2 renamed trapz -> trapezoid. Support both so the app runs either way.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# =============================================================================
# 1. CONFIGURATION — every fictional constant lives here
# =============================================================================

APP_TITLE = "SepsiSensor"
APP_SUBTITLE = "Circadian-Aware Antibiotic Concentration Simulator"
DISCLAIMER = "Educational simulation only — not for clinical use."

# Colour palette: navy / teal healthcare theme.
NAVY_DEEP = "#0B2545"
NAVY = "#134074"
BLUE = "#2E7DBE"
TEAL = "#0F8B8D"
TEAL_LIGHT = "#7DE2D1"
AMBER = "#E8720C"
RED = "#C42B3B"
GREEN = "#1B7F5C"
INK = "#1C2B3A"
PAPER = "#F4F8FC"

# --- Fictional model constants -----------------------------------------------
# SIM_VOLUME is an invented "simulation volume" that converts a dose in
# simulation units into a concentration in simulation concentration units (SCU).
# It is NOT a volume of distribution for any real medicine.
SIM_VOLUME = 40.0

# Integration step, in hours. Smaller = smoother curve, slower to compute.
DT_HOURS = 0.02

CONC_UNIT = "SCU"  # simulation concentration units

# Baseline clearance options, in invented "simulation clearance units per hour".
BASELINE_CLEARANCE = {"Low": 3.0, "Typical": 5.0, "High": 8.0}

# How strongly the clearance wave swings above and below baseline, as a fraction.
CIRCADIAN_AMPLITUDE = {"None": 0.00, "Mild": 0.10, "Moderate": 0.25, "Strong": 0.40}

# Fixed predefined values used when "simulate unstable patient" is switched on.
# These are arbitrary constants chosen for the simulation. They are not doses,
# not clinical settings, and they deliberately do NOT increase the dose.
EMERGENCY_PROFILE = {
    "clearance": BASELINE_CLEARANCE["Typical"],
    "interval_hours": 8.0,
    "circadian_amplitude": 0.0,
}

HOUR_LABELS = [f"{h:02d}:00" for h in range(24)]

DEFAULTS = {
    "duration": 48,
    "interval": 8,
    "dose": 1000,
    "clearance_level": "Typical",
    "circadian_strength": "Moderate",
    "start_clock": 8,
    "night_start": 23,
    "night_end": 7,
    "safety_range": (10.0, 40.0),
    "use_circadian": True,
    "unstable": False,
}

PRESETS = {
    "Balanced comparison": dict(DEFAULTS),
    "Accumulation scenario": {**DEFAULTS, "clearance_level": "Low",
                              "circadian_strength": "Strong", "interval": 6},
    "Under-range scenario": {**DEFAULTS, "clearance_level": "High", "interval": 12},
}

# --- Real-world data tab: dataset info ---------------------------------------
# This tab uses a REAL, published, de-identified research dataset. It is kept
# completely separate from the fictional pharmacokinetic simulator above: it
# contains no drug names and no dosing information, because legitimate public
# sepsis datasets do not include that — real per-patient drug and dose data is
# protected clinical information and is not released publicly.
REAL_DATA_NAME = "Sepsis Survival Minimal Clinical Records"
REAL_DATA_CITATION = ("Chicco, D. & Jurman, G. (2020). Sepsis Survival Minimal "
                      "Clinical Records. UCI Machine Learning Repository. "
                      "https://doi.org/10.24432/C53C8N")
REAL_DATA_PAPER = "https://www.nature.com/articles/s41598-020-73558-3"
REAL_DATA_UCI_PAGE = "https://archive.ics.uci.edu/dataset/827"
REAL_DATA_DOWNLOAD = "https://archive.ics.uci.edu/static/public/827/sepsis+survival+minimal+clinical+records.zip"
REAL_DATA_COLUMNS = ["age_years", "sex_0male_1female", "episode_number",
                    "hospital_outcome_1alive_0dead"]
REAL_DATA_COLUMN_ALIASES = {
    "age": "age_years", "age_years": "age_years",
    "sex": "sex_0male_1female", "gender": "sex_0male_1female",
    "sex_0male_1female": "sex_0male_1female",
    "episode": "episode_number", "episode_number": "episode_number",
    "prior_episodes": "episode_number",
    "outcome": "hospital_outcome_1alive_0dead",
    "hospital_outcome_1alive_0dead": "hospital_outcome_1alive_0dead",
    "survived": "hospital_outcome_1alive_0dead",
}


# =============================================================================
# 2. MODEL — pure maths, no Streamlit calls in this section
# =============================================================================

def clock_from_elapsed(elapsed_hours: np.ndarray, start_clock: float) -> np.ndarray:
    """Convert hours since the start of the run into a 24-hour clock time."""
    return (start_clock + elapsed_hours) % 24.0


def is_night(clock_hours: np.ndarray, night_start: float, night_end: float) -> np.ndarray:
    """Boolean mask for clock times inside the night window (handles midnight wrap)."""
    if night_start <= night_end:
        return (clock_hours >= night_start) & (clock_hours < night_end)
    return (clock_hours >= night_start) | (clock_hours < night_end)


def night_midpoint(night_start: float, night_end: float) -> float:
    """Middle of the night window, e.g. 23:00-07:00 gives 03:00."""
    length = (night_end - night_start) % 24.0 or 24.0
    return (night_start + length / 2.0) % 24.0


def clearance_at(clock_hours: np.ndarray, baseline: float,
                 amplitude: float, night_mid: float) -> np.ndarray:
    """
    Simulated clearance as a smooth 24-hour cosine wave.

        clearance(t) = baseline * (1 + amplitude * cos(2*pi*(clock - peak) / 24))

    The wave peaks 12 hours away from the middle of the night window, so the
    lowest simulated clearance falls in the middle of the night. With
    amplitude = 0 the wave flattens and clearance is constant.
    """
    peak_hour = (night_mid + 12.0) % 24.0
    wave = np.cos(2.0 * np.pi * (clock_hours - peak_hour) / 24.0)
    return baseline * (1.0 + amplitude * wave)


@dataclass
class Simulation:
    """One complete simulated scenario."""
    label: str
    colour: str
    time: np.ndarray           # hours since start
    clock: np.ndarray          # clock time at each step
    concentration: np.ndarray  # simulated concentration, SCU
    clearance: np.ndarray      # simulated clearance at each step
    dose_times: np.ndarray
    dose_values: np.ndarray


def run_simulation(label: str, colour: str, duration: float, interval: float,
                   dose: float, baseline_clearance: float, amplitude: float,
                   start_clock: float, night_mid: float) -> Simulation:
    """
    One-compartment simulation with a time-varying elimination rate.

    At each dosing time the concentration jumps by dose / SIM_VOLUME. Between
    doses it decays exponentially, using the local elimination rate for that
    moment of the day.
    """
    n_steps = int(round(duration / DT_HOURS)) + 1
    time = np.linspace(0.0, duration, n_steps)
    clock = clock_from_elapsed(time, start_clock)

    clearance = clearance_at(clock, baseline_clearance, amplitude, night_mid)
    k = clearance / SIM_VOLUME  # elimination rate constant, per hour

    dose_times = np.arange(0.0, duration - 1e-9, interval)
    dose_indices = {int(round(t / DT_HOURS)) for t in dose_times}
    jump = dose / SIM_VOLUME

    concentration = np.zeros(n_steps)
    current = 0.0
    for i in range(n_steps):
        if i in dose_indices:
            current += jump
        concentration[i] = current
        if i < n_steps - 1:
            k_mid = 0.5 * (k[i] + k[i + 1])          # midpoint rate for the step
            current *= float(np.exp(-k_mid * DT_HOURS))

    dose_values = np.array([concentration[int(round(t / DT_HOURS))] for t in dose_times])

    return Simulation(label=label, colour=colour, time=time, clock=clock,
                      concentration=concentration, clearance=clearance,
                      dose_times=dose_times, dose_values=dose_values)


def night_blocks(time: np.ndarray, clock: np.ndarray,
                 night_start: float, night_end: float) -> list[tuple[float, float]]:
    """Find contiguous stretches of simulation time that fall in the night window."""
    mask = is_night(clock, night_start, night_end)
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    return [(float(time[g[0]]), float(time[g[-1]])) for g in groups]


# =============================================================================
# 3. METRICS
# =============================================================================

@dataclass
class Metrics:
    cmax: float
    cmin: float
    hours_above: float
    hours_below: float
    auc: float
    alerts: int


def compute_metrics(sim: Simulation, lower: float, upper: float) -> Metrics:
    """
    Summarise one simulated curve.

    An 'alert event' is counted each time the curve crosses from inside the
    fictional safety range to outside it — not once per time step.
    """
    c = sim.concentration
    outside = (c > upper) | (c < lower)
    crossings = int(np.sum(outside[1:] & ~outside[:-1]))
    if outside[0]:
        crossings += 1

    return Metrics(
        cmax=float(c.max()),
        cmin=float(c.min()),
        hours_above=float(np.sum(c > upper) * DT_HOURS),
        hours_below=float(np.sum(c < lower) * DT_HOURS),
        auc=float(_trapezoid(c, sim.time)),
        alerts=crossings,
    )


def write_interpretation(base: Simulation, base_m: Metrics,
                         comp: Simulation, comp_m: Metrics,
                         unstable: bool, lower: float, upper: float) -> str:
    """Build a plain-English paragraph describing what the numbers did."""
    peak_diff = comp_m.cmax - base_m.cmax
    auc_diff = comp_m.auc - base_m.auc

    if abs(peak_diff) < 0.5:
        peak_text = ("produced a maximum predicted concentration almost identical to "
                     "the fixed schedule")
    elif peak_diff > 0:
        peak_text = (f"produced a maximum predicted concentration "
                     f"{peak_diff:.1f} {CONC_UNIT} higher than the fixed schedule")
    else:
        peak_text = (f"produced a maximum predicted concentration "
                     f"{abs(peak_diff):.1f} {CONC_UNIT} lower than the fixed schedule")

    if abs(auc_diff) < 1.0:
        auc_text = "Total simulated exposure was essentially unchanged."
    elif auc_diff > 0:
        auc_text = (f"Total simulated exposure rose by {auc_diff:.0f} "
                    f"{CONC_UNIT}·h over the run.")
    else:
        auc_text = (f"Total simulated exposure fell by {abs(auc_diff):.0f} "
                    f"{CONC_UNIT}·h over the run.")

    if comp_m.hours_above > 0.05:
        range_text = (f"The {comp.label.lower()} spent {comp_m.hours_above:.1f} h above the "
                      f"fictional upper threshold of {upper:.0f} {CONC_UNIT}")
    elif comp_m.hours_below > 0.05:
        range_text = (f"The {comp.label.lower()} spent {comp_m.hours_below:.1f} h below the "
                      f"fictional lower threshold of {lower:.0f} {CONC_UNIT}")
    else:
        range_text = (f"The {comp.label.lower()} stayed inside the fictional safety range "
                      f"for the whole run")

    if unstable:
        mode_text = ("Unstable-patient mode is on, so circadian adjustment was bypassed and "
                     "fixed predefined emergency simulation parameters were used instead. ")
    else:
        mode_text = ""

    return (
        f"{mode_text}In this fictional simulation, the {comp.label.lower()} {peak_text}. "
        f"{range_text}, compared with {base_m.hours_above:.1f} h above and "
        f"{base_m.hours_below:.1f} h below for the fixed schedule. {auc_text} "
        f"These are mathematical results produced by invented numbers. They are not a "
        f"treatment recommendation and they say nothing about what would happen to a real person."
    )


# =============================================================================
# 3b. REAL-WORLD DATA — loading, demo fallback, and a simple statistical model
# =============================================================================
# Everything in this section works with REAL, de-identified, published research
# data (or a clearly-labelled synthetic stand-in). It never touches the
# fictional pharmacokinetic model above, and it never outputs anything framed
# as a dose or a treatment decision — only a statistical association drawn
# from a specific historical dataset.

def parse_real_dataset(uploaded_bytes: bytes) -> tuple[pd.DataFrame | None, str | None]:
    """
    Read an uploaded CSV and map it onto the four expected columns.
    Returns (dataframe, None) on success or (None, error_message) on failure.
    """
    try:
        raw = pd.read_csv(io.BytesIO(uploaded_bytes))
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
        return None, f"Could not read that file as a CSV ({exc})."

    lowered = {c.strip().lower(): c for c in raw.columns}
    rename_map = {}
    for alias, target in REAL_DATA_COLUMN_ALIASES.items():
        if alias in lowered:
            rename_map[lowered[alias]] = target
    df = raw.rename(columns=rename_map)

    missing = [c for c in REAL_DATA_COLUMNS if c not in df.columns]
    if missing:
        return None, ("This file is missing expected column(s): "
                      f"{', '.join(missing)}. Expected columns (or common "
                      f"aliases like 'age', 'sex', 'outcome') are: "
                      f"{', '.join(REAL_DATA_COLUMNS)}.")

    df = df[REAL_DATA_COLUMNS].dropna()
    for col in REAL_DATA_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    if df.empty:
        return None, "No valid rows were found after checking the required columns."
    return df.reset_index(drop=True), None


def generate_demo_real_dataset(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """
    A small SYNTHETIC dataset shaped like the real one, for trying the tab out
    with no download required. This is NOT real patient data and its numbers
    are not drawn from the actual published study — it exists only so the
    charts and model below have something to run on before you load the
    genuine file.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 95, size=n)
    sex = rng.integers(0, 2, size=n)
    episode = rng.choice([1, 2, 3, 4], size=n, p=[0.7, 0.18, 0.08, 0.04])

    # Arbitrary synthetic logit — invented for demo purposes only.
    logit = -3.2 + 0.035 * age + 0.25 * episode - 0.1 * sex
    prob_death = 1 / (1 + np.exp(-logit))
    outcome_alive = (rng.random(n) > prob_death).astype(int)

    return pd.DataFrame({
        "age_years": age,
        "sex_0male_1female": sex,
        "episode_number": episode,
        "hospital_outcome_1alive_0dead": outcome_alive,
    })


@dataclass
class RealDataModel:
    accuracy: float
    auc: float
    coefficients: dict[str, float]
    n_train: int
    n_test: int
    model: LogisticRegression


def fit_real_data_model(df: pd.DataFrame) -> RealDataModel | None:
    """Fit a simple logistic regression predicting survival from the three features."""
    features = ["age_years", "sex_0male_1female", "episode_number"]
    X = df[features].values
    y = df["hospital_outcome_1alive_0dead"].values

    if len(np.unique(y)) < 2 or len(df) < 20:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, preds))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auc = float("nan")

    coefficients = dict(zip(features, model.coef_[0].tolist()))
    return RealDataModel(accuracy=accuracy, auc=auc, coefficients=coefficients,
                         n_train=len(X_train), n_test=len(X_test), model=model)


# =============================================================================
# 4. USER INTERFACE
# =============================================================================

st.set_page_config(page_title=f"{APP_TITLE} — {APP_SUBTITLE}",
                   page_icon="🩺", layout="wide")

# --- Styling ------------------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{ background: {PAPER}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
html, body, [class*="css"] {{ color: {INK}; }}

.sepsi-header {{
  background: linear-gradient(120deg, {NAVY_DEEP} 0%, {NAVY} 55%, {TEAL} 140%);
  border-radius: 14px; padding: 1.5rem 1.8rem; color: #fff;
  border-left: 7px solid {TEAL_LIGHT};
}}
.sepsi-header h1 {{ margin: 0; font-size: 2.35rem; letter-spacing: -0.5px; color: #fff; }}
.sepsi-header p {{ margin: .35rem 0 0; font-size: 1.05rem; color: {TEAL_LIGHT}; }}

.sepsi-warning {{
  background: #FFF1E6; border: 2px solid {AMBER}; border-left: 7px solid {RED};
  color: #6B2708; border-radius: 10px; padding: .85rem 1.1rem; margin: .9rem 0 1.3rem;
  font-weight: 600; font-size: 1.02rem;
}}

.card {{
  background: #fff; border: 1px solid #DCE6F0; border-top: 4px solid {TEAL};
  border-radius: 11px; padding: .95rem 1.05rem; height: 100%;
}}
.card .label {{ font-size: .82rem; color: #5C7189; margin-bottom: .25rem; }}
.card .value {{ font-size: 1.75rem; font-weight: 700; color: {NAVY_DEEP}; line-height: 1.15; }}
.card .unit {{ font-size: .8rem; color: #5C7189; }}

.status {{ border-radius: 12px; padding: 1.2rem 1.4rem; font-size: 1.05rem; }}
.status h3 {{ margin: 0 0 .4rem; font-size: 1.3rem; }}
.status-ok {{ background: #E8F6EF; border: 2px solid {GREEN}; color: #0E4C36; }}
.status-alert {{ background: #FDECEA; border: 2px solid {RED}; color: #6B1017; }}

.note {{
  background: #EAF2FA; border-left: 5px solid {BLUE}; border-radius: 8px;
  padding: .9rem 1.1rem; margin: .6rem 0;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
  background: #E4EDF7; border-radius: 9px 9px 0 0; padding: .55rem 1.05rem;
  font-weight: 600; color: {NAVY};
}}
.stTabs [aria-selected="true"] {{ background: {NAVY}; color: #fff; }}
</style>
""", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render HTML safely: collapse to one line so Streamlit never sees indentation."""
    st.markdown(" ".join(line.strip() for line in markup.strip().splitlines()),
                unsafe_allow_html=True)


def metric_card(label: str, value: str, unit: str = "") -> None:
    html(f'<div class="card"><div class="label">{label}</div>'
         f'<div class="value">{value}</div><div class="unit">{unit}</div></div>')


# --- Header and permanent warning banner --------------------------------------
html(f'<div class="sepsi-header"><h1>{APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>')
html(f'<div class="sepsi-warning">⚠️ {DISCLAIMER} This tool does not diagnose, '
     f'treat or monitor anyone. All values are invented.</div>')

# --- Session state ------------------------------------------------------------
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.caption(DISCLAIMER)
    st.markdown("**Quick scenarios**")
    for name in PRESETS:
        st.button(name, use_container_width=True, on_click=apply_preset, args=(name,))
    st.divider()
    st.markdown("**What this project asks**")
    st.write("If a mathematical model lets simulated clearance rise and fall over a "
             "24-hour cycle, how does the predicted concentration curve differ from "
             "one with constant clearance?")
    st.divider()
    st.caption("Fictional constants: simulation volume = "
               f"{SIM_VOLUME:.0f} volume units · step size = {DT_HOURS} h")

tabs = st.tabs(["Overview", "Simulation Controls", "Results",
                "Emergency Override", "How It Works", "Limitations",
                "Real-World Data"])

# -----------------------------------------------------------------------------
# TAB 1 — Overview
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("What this project investigates")
    st.write(
        "Sepsis is a serious condition where the body's response to an infection starts "
        "to damage its own tissues and organs. Antibiotics are one part of hospital care "
        "for it, and how much of a medicine stays in the bloodstream over time matters. "
        "Some antibiotics may carry kidney-related toxicity risks, particularly if they "
        "accumulate, while concentrations that stay too low may be less effective. Real "
        "decisions about any of this are made by clinicians using laboratory results, "
        "patient characteristics and validated medical guidelines."
    )
    st.write(
        "Separately, the body runs on a circadian rhythm — an internal clock of roughly "
        "24 hours. Researchers have asked whether processes that remove substances from "
        "the blood might also follow a daily pattern. **SepsiSensor** does not answer that "
        "question. It builds a simple mathematical model where clearance is *allowed* to "
        "vary over the day, and shows what that assumption does to a predicted curve."
    )

    left, right = st.columns(2)
    with left:
        html(f"""
        <div class="note"><strong>The two scenarios compared</strong><br>
        <strong>1. Fixed schedule.</strong> Doses at a constant interval, clearance held constant
        all day.<br>
        <strong>2. Circadian-aware model.</strong> Identical doses, but simulated clearance rises
        and falls as a smooth 24-hour wave.</div>
        """)
    with right:
        html(f"""
        <div class="note"><strong>What is measured</strong><br>
        Peak and trough predicted concentration, hours spent outside a fictional safety range,
        total simulated exposure (AUC), and the number of simulated alert events — all in
        invented simulation units, over {DEFAULTS['duration']} simulated hours.</div>
        """)

    st.markdown("#### What this is not")
    st.markdown(
        "- It is **not** a medical device and **not** clinical decision support.\n"
        "- It **cannot** calculate a dose. Doses here are in invented *simulation units*, never mg.\n"
        "- It does **not** treat sepsis, save lives, prevent kidney damage, or belong in a hospital.\n"
        "- It uses **no** real patient data and models **no** real medicine."
    )

# -----------------------------------------------------------------------------
# TAB 2 — Simulation Controls
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Set up the simulated scenario")
    st.caption("Every value below is fictional and chosen to make the maths visible on a chart.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Timing**")
        duration = st.slider("Simulation duration (hours)", 24, 96,
                             key="duration", step=12)
        interval = st.slider("Fixed dosing interval (hours)", 4, 24,
                             key="interval", step=1)
        start_clock = st.selectbox("Clock time at hour 0", options=list(range(24)),
                                   format_func=lambda h: HOUR_LABELS[h], key="start_clock")

    with col_b:
        st.markdown("**Simulated pharmacokinetics**")
        dose = st.slider("Dose per administration (simulation units)", 200, 2000,
                         key="dose", step=50,
                         help="A fictional quantity. Not milligrams, not any real medicine.")
        clearance_level = st.selectbox("Baseline clearance", list(BASELINE_CLEARANCE),
                                       key="clearance_level")
        circadian_strength = st.selectbox("Circadian effect strength",
                                          list(CIRCADIAN_AMPLITUDE), key="circadian_strength")

    with col_c:
        st.markdown("**Night window and safety range**")
        night_start = st.selectbox("Night-time starts", options=list(range(24)),
                                   format_func=lambda h: HOUR_LABELS[h], key="night_start")
        night_end = st.selectbox("Night-time ends", options=list(range(24)),
                                 format_func=lambda h: HOUR_LABELS[h], key="night_end")
        safety_range = st.slider(f"Simulated safety range ({CONC_UNIT})",
                                 0.0, 90.0, key="safety_range", step=1.0,
                                 help="Fictional lower and upper thresholds used only for "
                                      "comparison between the two models.")

    st.divider()
    tog_a, tog_b = st.columns(2)
    with tog_a:
        use_circadian = st.toggle("Enable circadian-aware clearance model", key="use_circadian")
    with tog_b:
        unstable = st.toggle("Simulate unstable patient", key="unstable")
        if unstable:
            st.caption("Circadian adjustment is bypassed. See the Emergency Override tab.")

    lower, upper = safety_range
    night_mid = night_midpoint(night_start, night_end)
    amplitude = CIRCADIAN_AMPLITUDE[circadian_strength]
    baseline = BASELINE_CLEARANCE[clearance_level]

    st.markdown("**Current configuration**")
    config_rows = [
        ("Duration", f"{duration} h"),
        ("Dosing interval", f"{interval} h"),
        ("Dose", f"{dose} simulation units"),
        ("Baseline clearance", f"{clearance_level} ({baseline:.1f} units/h)"),
        ("Circadian strength", f"{circadian_strength} (±{amplitude*100:.0f}%)"),
        ("Night window", f"{HOUR_LABELS[night_start]}–{HOUR_LABELS[night_end]} "
                         f"(middle {night_mid:04.1f}h)"),
        ("Safety range", f"{lower:.0f}–{upper:.0f} {CONC_UNIT}"),
        ("Mode", "Emergency fixed parameters" if unstable
                 else ("Circadian-aware active" if use_circadian else "Fixed schedule only")),
    ]
    st.dataframe(pd.DataFrame(config_rows, columns=["Setting", "Value"]),
                 hide_index=True, use_container_width=True)

# --- Build the simulations (shared by the tabs below) -------------------------
baseline_sim = run_simulation(
    label="Fixed schedule", colour=BLUE, duration=duration, interval=interval,
    dose=dose, baseline_clearance=baseline, amplitude=0.0,
    start_clock=start_clock, night_mid=night_mid,
)

if unstable:
    comparison_sim = run_simulation(
        label="Emergency fixed-parameter model", colour=AMBER, duration=duration,
        interval=EMERGENCY_PROFILE["interval_hours"], dose=dose,
        baseline_clearance=EMERGENCY_PROFILE["clearance"],
        amplitude=EMERGENCY_PROFILE["circadian_amplitude"],
        start_clock=start_clock, night_mid=night_mid,
    )
elif use_circadian:
    comparison_sim = run_simulation(
        label="Circadian-aware model", colour=TEAL, duration=duration, interval=interval,
        dose=dose, baseline_clearance=baseline, amplitude=amplitude,
        start_clock=start_clock, night_mid=night_mid,
    )
else:
    comparison_sim = None

baseline_metrics = compute_metrics(baseline_sim, lower, upper)
comparison_metrics = compute_metrics(comparison_sim, lower, upper) if comparison_sim else None
shaded_nights = night_blocks(baseline_sim.time, baseline_sim.clock, night_start, night_end)


def clock_labels(clock: np.ndarray) -> list[str]:
    return [f"{int(h):02d}:{int(round((h % 1) * 60)):02d}" for h in clock]


def add_night_shading(fig: go.Figure) -> None:
    for i, (t0, t1) in enumerate(shaded_nights):
        fig.add_vrect(x0=t0, x1=t1, fillcolor=NAVY_DEEP, opacity=0.07, line_width=0,
                      layer="below",
                      annotation_text="night" if i == 0 else None,
                      annotation_position="top left",
                      annotation_font_size=10, annotation_font_color="#5C7189")


def add_trace(fig: go.Figure, sim: Simulation, dash: str | None = None) -> None:
    fig.add_trace(go.Scatter(
        x=sim.time, y=sim.concentration, name=sim.label, mode="lines",
        line=dict(color=sim.colour, width=3, dash=dash),
        customdata=clock_labels(sim.clock),
        hovertemplate=("Hour %{x:.1f} (%{customdata})<br>"
                       f"Concentration %{{y:.1f}} {CONC_UNIT}<extra>{sim.label}</extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=sim.dose_times, y=sim.dose_values, name=f"{sim.label} — dosing events",
        mode="markers", marker=dict(symbol="triangle-up", size=11, color=sim.colour,
                                    line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate=("Simulated dosing event<br>Hour %{x:.0f}"
                       f"<extra>{sim.label}</extra>"),
        showlegend=True,
    ))


# -----------------------------------------------------------------------------
# TAB 3 — Results
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Predicted concentration over time")

    fig = go.Figure()
    add_night_shading(fig)
    fig.add_hrect(y0=lower, y1=upper, fillcolor=TEAL_LIGHT, opacity=0.22,
                  line_width=0, layer="below",
                  annotation_text="simulated safety range",
                  annotation_position="top right",
                  annotation_font_size=11, annotation_font_color=TEAL)
    fig.add_hline(y=upper, line=dict(color=RED, width=1.4, dash="dot"))
    fig.add_hline(y=lower, line=dict(color=AMBER, width=1.4, dash="dot"))

    add_trace(fig, baseline_sim)
    if comparison_sim:
        add_trace(fig, comparison_sim, dash="dash" if unstable else None)

    ticks = np.arange(0, duration + 1, 6)
    tick_text = [f"{int(t)}h<br>{HOUR_LABELS[int((start_clock + t) % 24)]}" for t in ticks]

    fig.update_layout(
        title=dict(text=f"Simulated concentration — {DISCLAIMER}",
                   font=dict(size=15, color=NAVY_DEEP)),
        xaxis=dict(title="Hours since start of simulation", tickmode="array",
                   tickvals=ticks, ticktext=tick_text, gridcolor="#E2EAF3"),
        yaxis=dict(title=f"Predicted concentration ({CONC_UNIT})", gridcolor="#E2EAF3",
                   rangemode="tozero"),
        hovermode="x unified", height=520, plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF", margin=dict(t=60, b=50, l=60, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Metric cards ---------------------------------------------------------
    focus = comparison_sim or baseline_sim
    focus_metrics = comparison_metrics or baseline_metrics
    st.markdown(f"**Summary metrics — {focus.label}**")

    r1 = st.columns(3)
    with r1[0]:
        metric_card("Maximum predicted concentration", f"{focus_metrics.cmax:.1f}", CONC_UNIT)
    with r1[1]:
        metric_card("Minimum predicted concentration", f"{focus_metrics.cmin:.1f}", CONC_UNIT)
    with r1[2]:
        metric_card("Total simulated exposure (AUC)", f"{focus_metrics.auc:.0f}",
                    f"{CONC_UNIT}·h")
    st.write("")
    r2 = st.columns(3)
    with r2[0]:
        metric_card("Time above simulated upper threshold",
                    f"{focus_metrics.hours_above:.1f}", "hours")
    with r2[1]:
        metric_card("Time below simulated lower threshold",
                    f"{focus_metrics.hours_below:.1f}", "hours")
    with r2[2]:
        metric_card("Simulated alert events", f"{focus_metrics.alerts}",
                    "range crossings")

    st.caption(DISCLAIMER)
    st.divider()

    # --- Comparison table -----------------------------------------------------
    st.markdown("**Model comparison**")
    rows = [(baseline_sim.label, baseline_metrics)]
    if comparison_sim:
        rows.append((comparison_sim.label, comparison_metrics))

    table = pd.DataFrame([{
        "Model": name,
        f"Maximum concentration ({CONC_UNIT})": round(m.cmax, 1),
        f"Minimum concentration ({CONC_UNIT})": round(m.cmin, 1),
        "Time above range (h)": round(m.hours_above, 1),
        "Time below range (h)": round(m.hours_below, 1),
        f"AUC ({CONC_UNIT}·h)": round(m.auc, 0),
        "Alerts": m.alerts,
    } for name, m in rows])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(DISCLAIMER)

    # --- Interpretation -------------------------------------------------------
    st.markdown("**Plain-English interpretation**")
    if comparison_sim:
        html(f'<div class="note">{write_interpretation(baseline_sim, baseline_metrics, comparison_sim, comparison_metrics, unstable, lower, upper)}</div>')
    else:
        html('<div class="note">Only the fixed schedule is running. Switch on the '
             'circadian-aware clearance model in Simulation Controls to compare two curves.</div>')

    # --- Clearance wave -------------------------------------------------------
    st.divider()
    st.markdown("**Simulated clearance across the day**")
    fig_cl = go.Figure()
    add_night_shading(fig_cl)
    fig_cl.add_trace(go.Scatter(x=baseline_sim.time, y=baseline_sim.clearance,
                                name="Constant clearance", mode="lines",
                                line=dict(color=BLUE, width=2.5)))
    if comparison_sim:
        fig_cl.add_trace(go.Scatter(x=comparison_sim.time, y=comparison_sim.clearance,
                                    name=comparison_sim.label, mode="lines",
                                    line=dict(color=comparison_sim.colour, width=2.5)))
    fig_cl.update_layout(
        title=dict(text=f"Simulated clearance input — {DISCLAIMER}",
                   font=dict(size=14, color=NAVY_DEEP)),
        xaxis=dict(title="Hours since start", tickmode="array", tickvals=ticks,
                   ticktext=tick_text, gridcolor="#E2EAF3"),
        yaxis=dict(title="Clearance (simulation units/h)", gridcolor="#E2EAF3"),
        height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=55, b=40, l=60, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    st.plotly_chart(fig_cl, use_container_width=True)

    # --- Data export ----------------------------------------------------------
    export = pd.DataFrame({
        "hours_since_start": baseline_sim.time,
        "clock_time": clock_labels(baseline_sim.clock),
        f"fixed_schedule_{CONC_UNIT}": baseline_sim.concentration,
    })
    if comparison_sim:
        export[f"{comparison_sim.label.lower().replace(' ', '_')}_{CONC_UNIT}"] = \
            comparison_sim.concentration
    st.download_button("Download simulated data (CSV)",
                       data=export.to_csv(index=False).encode("utf-8"),
                       file_name="sepsisensor_simulated_data.csv", mime="text/csv")
    st.caption("Useful for tables and graphs in a written project report. "
               f"{DISCLAIMER}")

# -----------------------------------------------------------------------------
# TAB 4 — Emergency Override
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Emergency override behaviour")

    if not unstable:
        html(f"""
        <div class="status status-ok"><h3>✅ Stable simulated scenario: circadian model active</h3>
        The simulation is running its normal comparison. Simulated clearance follows the
        24-hour wave you configured, and both curves use the dosing interval you set.</div>
        """)
    else:
        html(f"""
        <div class="status status-alert"><h3>🟠 Emergency simulation mode: circadian adjustment
        bypassed. Fixed emergency parameters active.</h3>
        The circadian wave has been switched off. The model now runs on fixed predefined
        simulation parameters instead of time-of-day adjustment.</div>
        """)

    st.write("")
    st.markdown("**Why a model would do this**")
    st.write(
        "A model that adjusts itself based on time of day is making an assumption: that the "
        "situation is steady enough for a daily pattern to be meaningful. If the simulated "
        "scenario is described as unstable, that assumption no longer holds. Rather than keep "
        "adjusting on an assumption that may not apply, the simulation falls back to fixed, "
        "predictable, predefined values. This represents prioritising immediate emergency "
        "modelling over time-of-day optimisation."
    )
    st.write(
        "This is a design principle for software, not a statement about patient care. Deciding "
        "what happens in an emergency is the job of clinicians and validated medical guidelines."
    )

    st.markdown("**Fixed predefined emergency simulation parameters**")
    st.dataframe(pd.DataFrame([
        ("Circadian adjustment", "Off (amplitude fixed at 0)"),
        ("Simulated clearance", f"{EMERGENCY_PROFILE['clearance']:.1f} simulation units/h"),
        ("Dosing interval", f"{EMERGENCY_PROFILE['interval_hours']:.0f} h"),
        ("Dose per administration", "Unchanged from your setting"),
    ], columns=["Parameter", "Fixed predefined value"]), hide_index=True,
        use_container_width=True)

    html(f"""
    <div class="note"><strong>Read this carefully.</strong> These are arbitrary numbers written
    into the source code so the fallback behaviour can be demonstrated. The override does not
    increase the dose and does not calculate anything for a person. It changes which fictional
    constants the maths uses. {DISCLAIMER}</div>
    """)

# -----------------------------------------------------------------------------
# TAB 5 — How It Works
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("The ideas behind the model")

    st.markdown("**Clearance**")
    st.write("In this model, clearance is a number describing how quickly the simulated "
             "substance is removed from a simulated bloodstream. A bigger clearance means "
             "the concentration falls faster between doses.")

    st.markdown("**Circadian rhythm**")
    st.write("The body's internal clock runs on a cycle of roughly 24 hours and influences "
             "things like sleep, temperature and hormone levels. This project asks a "
             "*what-if*: if clearance also followed a daily cycle, what would the maths look "
             "like? The model uses a smooth cosine wave with its lowest point in the middle "
             "of the night window.")

    st.markdown("**Drug concentration**")
    st.write("The simulated amount of substance in the bloodstream at each moment. Each dose "
             "makes it jump up; between doses it decays.")

    st.markdown("**Safety range**")
    st.write("Two fictional numbers — an upper and a lower limit. They exist so the two "
             "models can be compared on the same yardstick. They are not derived from any "
             "medical source and mean nothing outside this simulation.")

    st.divider()
    st.markdown("**The maths, step by step**")
    st.markdown(
        "1. The body is treated as a **single container** of fixed simulated volume.\n"
        "2. At each dosing time the concentration **jumps** by `dose ÷ simulation volume`.\n"
        "3. Between doses the concentration **decays exponentially**, "
        "`C(t + Δt) = C(t) × e^(−k·Δt)`.\n"
        "4. The elimination rate is `k = clearance ÷ simulation volume`.\n"
        "5. In the **fixed** model, `k` never changes.\n"
        "6. In the **circadian-aware** model, clearance follows "
        "`baseline × (1 + amplitude × cos(2π(clock − peak) ÷ 24))`, so `k` changes hour by hour.\n"
        "7. The simulation steps forward in small increments of "
        f"{DT_HOURS} h and records the concentration at every step."
    )

    st.markdown("**Why the night matters in this model**")
    st.write("The wave is built so its lowest clearance sits in the middle of your night "
             "window. Lower clearance means slower decay, so the concentration falls less "
             "between night-time doses and the next peak starts from higher up. That is why "
             "the teal curve can drift above the blue one overnight. It is arithmetic, not a "
             "discovery about human biology.")

    st.markdown("**What varies, what stays fixed**")
    st.dataframe(pd.DataFrame([
        ("Dose amount", "Set by you", "Identical in both models"),
        ("Dosing interval", "Set by you", "Identical unless emergency mode is on"),
        ("Simulated volume", f"{SIM_VOLUME:.0f} volume units", "Fixed constant in the code"),
        ("Clearance", "Baseline set by you", "Constant in fixed model, wave in circadian model"),
        ("Safety thresholds", "Set by you", "Used only for measurement, never fed into the maths"),
    ], columns=["Quantity", "Where it comes from", "Role"]), hide_index=True,
        use_container_width=True)

    st.caption("The project compares mathematical scenarios, not real people. " + DISCLAIMER)

# -----------------------------------------------------------------------------
# TAB 6 — Limitations
# -----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Limitations")
    html(f'<div class="sepsi-warning">⚠️ {DISCLAIMER}</div>')

    st.markdown(
        "- **This is a simplified mathematical model.** A single container with one decay "
        "rate is the simplest pharmacokinetic idea there is. Real pharmacokinetics uses "
        "multiple compartments, absorption, protein binding and far more.\n"
        "- **It does not use real patient data.** No records, no measurements, no real people.\n"
        "- **It models no real medicine.** There is no drug name anywhere in this project, and "
        "the units are invented.\n"
        "- **It leaves out nearly everything that matters in sepsis treatment**, including age, "
        "weight, infection source, organ function, kidney and liver performance, blood test "
        "results, other medicines, allergies, resistance patterns and clinician judgement.\n"
        "- **It cannot diagnose sepsis, prescribe treatment, or predict outcomes** for any real "
        "person, and it is not suitable for hospital use.\n"
        "- **The circadian assumption is an input, not a finding.** The size of the daily swing "
        "is a dial this app lets you turn. The app cannot tell you what value is correct.\n"
        "- **The safety thresholds are invented.** They were chosen to make differences visible "
        "on a chart.\n"
        "- **Alert counts depend on the step size.** A curve that hovers exactly on a threshold "
        "can register extra crossings."
    )

    st.markdown("**What it is for**")
    st.write(
        "Learning. It supports understanding of three things: how a basic pharmacokinetic "
        "model behaves, what a circadian rhythm is, and how health technology should be "
        "designed responsibly — with limits stated plainly, assumptions exposed as adjustable "
        "inputs, and a fallback that becomes simpler rather than more aggressive when the "
        "scenario is uncertain."
    )

    st.markdown("**If you want to extend the project**")
    st.markdown(
        "- Run a sensitivity analysis: sweep circadian strength from none to strong and plot "
        "AUC against it.\n"
        "- Shift the night window in 2-hour steps and record how peak concentration moves.\n"
        "- Add a second simulated compartment and compare the curve shape.\n"
        "- Add uncertainty: run the model many times with slightly randomised clearance and "
        "plot a band instead of a line."
    )

# -----------------------------------------------------------------------------
# TAB 7 — Real-World Data
# -----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Explore a real, published sepsis dataset")

    html(f"""
    <div class="sepsi-warning">⚠️ This tab uses real, de-identified, published research
    data — but its output is a statistical pattern in one historical dataset, not a
    diagnosis, a risk score for any real person, or a treatment recommendation. It shares
    no code or numbers with the fictional simulator in the other tabs, and it contains no
    drug or dosing information of any kind.</div>
    """)

    st.write(
        f"**{REAL_DATA_NAME}** is a real dataset of {110204:,} hospital admissions from "
        "Norway (2011–2012), published in *Scientific Reports* and freely available "
        "under a CC BY 4.0 licence. It records three simple facts about each admission — "
        "age, sex, and how many prior sepsis episodes the patient had — plus whether they "
        "survived. It contains no medication, dose, or treatment information; public "
        "sepsis datasets generally don't, since that is protected clinical data."
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Citation:** {REAL_DATA_CITATION}")
        st.markdown(f"[Read the paper]({REAL_DATA_PAPER}) · "
                    f"[Dataset page]({REAL_DATA_UCI_PAGE}) · "
                    f"[Direct download]({REAL_DATA_DOWNLOAD})")
    with right:
        st.caption("Expected columns after unzipping: age_years, sex_0male_1female, "
                   "episode_number, hospital_outcome_1alive_0dead — this app also "
                   "recognises common alternates like 'age', 'sex', 'outcome'.")

    st.divider()
    st.markdown("**Load data**")
    uploaded = st.file_uploader("Upload the real dataset's CSV (or your own, same columns)",
                                type=["csv"])
    use_demo = st.toggle("Use synthetic demo data instead (no download needed)",
                         value=uploaded is None)

    real_df, load_error = None, None
    if uploaded is not None and not use_demo:
        real_df, load_error = parse_real_dataset(uploaded.getvalue())
        if load_error:
            st.error(load_error)
    if real_df is None:
        real_df = generate_demo_real_dataset()
        if uploaded is None or use_demo:
            st.info("Showing synthetic demo data shaped like the real dataset. Download "
                    "the real file above and upload it to see genuine results.", icon="ℹ️")

    st.dataframe(real_df.head(10), hide_index=True, use_container_width=True)
    st.caption(f"{len(real_df):,} rows loaded.")

    # --- Descriptive charts ----------------------------------------------------
    st.divider()
    st.markdown("**Patterns in this data**")

    survival_rate = real_df["hospital_outcome_1alive_0dead"].mean()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Records loaded", f"{len(real_df):,}")
    with c2:
        metric_card("Overall survival rate", f"{survival_rate*100:.1f}", "%")
    with c3:
        metric_card("Average age", f"{real_df['age_years'].mean():.0f}", "years")

    age_bins = pd.cut(real_df["age_years"], bins=range(0, 101, 10), right=False)
    by_age = real_df.groupby(age_bins, observed=True)["hospital_outcome_1alive_0dead"].mean()
    fig_age = go.Figure(go.Bar(
        x=[str(i) for i in by_age.index], y=by_age.values * 100,
        marker_color=TEAL,
        hovertemplate="Age %{x}<br>Survival rate %{y:.1f}%<extra></extra>",
    ))
    fig_age.update_layout(
        title=dict(text="Survival rate by age group", font=dict(size=14, color=NAVY_DEEP)),
        xaxis_title="Age group (years)", yaxis_title="Survival rate (%)",
        height=340, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=50, b=40, l=50, r=20), yaxis=dict(range=[0, 100], gridcolor="#E2EAF3"),
    )
    st.plotly_chart(fig_age, use_container_width=True)

    col_sex, col_ep = st.columns(2)
    with col_sex:
        by_sex = real_df.groupby("sex_0male_1female", observed=True)[
            "hospital_outcome_1alive_0dead"].mean()
        labels = {0: "Male", 1: "Female"}
        fig_sex = go.Figure(go.Bar(
            x=[labels.get(i, str(i)) for i in by_sex.index], y=by_sex.values * 100,
            marker_color=BLUE,
            hovertemplate="%{x}<br>Survival rate %{y:.1f}%<extra></extra>",
        ))
        fig_sex.update_layout(
            title=dict(text="Survival rate by sex", font=dict(size=13, color=NAVY_DEEP)),
            yaxis_title="Survival rate (%)", height=300, plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF", margin=dict(t=45, b=35, l=45, r=15),
            yaxis=dict(range=[0, 100], gridcolor="#E2EAF3"),
        )
        st.plotly_chart(fig_sex, use_container_width=True)
    with col_ep:
        by_ep = real_df.groupby("episode_number", observed=True)[
            "hospital_outcome_1alive_0dead"].mean()
        fig_ep = go.Figure(go.Bar(
            x=[str(i) for i in by_ep.index], y=by_ep.values * 100, marker_color=NAVY,
            hovertemplate="Episode #%{x}<br>Survival rate %{y:.1f}%<extra></extra>",
        ))
        fig_ep.update_layout(
            title=dict(text="Survival rate by prior episode count",
                       font=dict(size=13, color=NAVY_DEEP)),
            xaxis_title="Prior sepsis episodes", yaxis_title="Survival rate (%)",
            height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            margin=dict(t=45, b=35, l=45, r=15), yaxis=dict(range=[0, 100], gridcolor="#E2EAF3"),
        )
        st.plotly_chart(fig_ep, use_container_width=True)

    # --- Simple statistical model -----------------------------------------------
    st.divider()
    st.markdown("**A simple statistical model (logistic regression)**")
    st.caption("Trained on the data loaded above, split 75% train / 25% test.")

    fitted = fit_real_data_model(real_df)
    if fitted is None:
        st.warning("Not enough data (or only one outcome present) to fit a model here.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("Test-set accuracy", f"{fitted.accuracy*100:.1f}", "%")
        with m2:
            metric_card("Test-set AUC", f"{fitted.auc:.2f}" if fitted.auc == fitted.auc else "n/a")
        with m3:
            metric_card("Rows used", f"{fitted.n_train:,} train / {fitted.n_test:,} test")

        direction = {k: ("higher" if v > 0 else "lower") for k, v in fitted.coefficients.items()}
        html(f"""
        <div class="note">In this dataset, the model found that higher <strong>age</strong>
        was associated with a {direction['age_years']} chance of survival, being
        <strong>female</strong> (sex = 1) was associated with a {direction['sex_0male_1female']}
        chance of survival, and a higher <strong>prior episode count</strong> was associated
        with a {direction['episode_number']} chance of survival — all relative to the rest of
        this specific historical dataset. This describes a pattern in past data. It is not a
        diagnosis, and it does not predict what will happen to any real person.</div>
        """)

        st.markdown("**Try it (statistical illustration only)**")
        st.caption("This shows what the model above would output for a hypothetical "
                   "combination of the three factors. It is not a real risk assessment.")
        p1, p2, p3 = st.columns(3)
        with p1:
            try_age = st.slider("Age", 18, 95, 60, key="real_try_age")
        with p2:
            try_sex = st.selectbox("Sex", options=[0, 1],
                                   format_func=lambda v: "Male" if v == 0 else "Female",
                                   key="real_try_sex")
        with p3:
            try_ep = st.slider("Prior sepsis episodes", 1, 4, 1, key="real_try_ep")

        prob_alive = fitted.model.predict_proba([[try_age, try_sex, try_ep]])[0, 1]
        html(f"""
        <div class="note"><strong>Model output: {prob_alive*100:.0f}%</strong> predicted
        probability of survival for this combination, according to patterns in the loaded
        dataset only. This is a statistical illustration of the model above — never a real
        risk assessment, and never a basis for any decision about an actual patient.</div>
        """)

    st.caption(DISCLAIMER)

st.divider()
st.caption(f"{APP_TITLE} — {APP_SUBTITLE}. {DISCLAIMER} "
           "Real treatment decisions depend on clinicians, laboratory results, patient "
           "characteristics and validated medical guidelines.")
