# SepsiSensor

**A Circadian-Aware Simulation Engine for Safer Sepsis Care**

Science project — Health and Wellbeing, Intermediate level.

> ⚠️ **Educational simulation only — not for clinical use.**
> This program is not a medical device, not clinical decision support, and not a dose
> calculator. It uses invented numbers in fictional "simulation units". It cannot
> diagnose sepsis, recommend treatment, or predict what would happen to any real
> person. Real treatment decisions depend on clinicians, laboratory results, patient
> characteristics and validated medical guidelines.

---

## What it does

SepsiSensor compares two mathematical scenarios over a simulated 48-hour period:

| Scenario | Dosing | Clearance |
|---|---|---|
| Fixed schedule | Constant interval | Constant all day |
| Circadian-aware model | Same constant interval | Rises and falls as a 24-hour wave |

It then measures, for both, the peak and trough predicted concentration, the hours
spent outside a fictional safety range, the total simulated exposure (AUC) and the
number of simulated alert events.

A third mode — **simulate unstable patient** — switches the circadian wave off entirely
and falls back to fixed predefined emergency simulation parameters. It does **not**
increase the dose.

---

## Setup

You need Python 3.9 or newer.

### 1. Put the files in one folder

```
sepsisensor/
├── app.py
├── requirements.txt
└── README.md
```

### 2. Open the folder in VS Code

`File → Open Folder…` and choose `sepsisensor`.

### 3. Open a terminal inside VS Code

`Terminal → New Terminal` (or `` Ctrl+` ``). It opens already inside your project folder.

### 4. Create a virtual environment (recommended, not required)

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of the terminal line.

### 5. Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. To stop the app, press `Ctrl+C` in the
terminal.

---

## Using it

1. **Overview** — what the project asks and what it deliberately is not.
2. **Simulation Controls** — set duration, dosing interval, dose, baseline clearance,
   circadian strength, night window and the fictional safety range. Two toggles switch
   the circadian model and the unstable-patient scenario on and off.
3. **Results** — the main chart, six metric cards, the comparison table, a written
   interpretation, the clearance wave, and a CSV download of the raw simulated data.
4. **Emergency Override** — a status card showing which mode is active, and the exact
   fixed parameters used in emergency mode.
5. **How It Works** — the model explained for a student audience.
6. **Limitations** — what the project cannot do.
7. **Real-World Data** — a completely separate section using a real, published sepsis
   dataset (see below). It shares no code and no numbers with the fictional simulator in
   tabs 1–6.

---

## The Real-World Data tab

This tab is built on the **Sepsis Survival Minimal Clinical Records** dataset: 110,204 real
hospital admissions from Norway (2011–2012), published in *Scientific Reports*, free to
download under a CC BY 4.0 licence, no credentialing required.

- **Paper:** https://www.nature.com/articles/s41598-020-73558-3
- **Dataset page:** https://archive.ics.uci.edu/dataset/827
- **Direct download:** https://archive.ics.uci.edu/static/public/827/sepsis+survival+minimal+clinical+records.zip
- **Citation:** Chicco, D. & Jurman, G. (2020). Sepsis Survival Minimal Clinical Records.
  UCI Machine Learning Repository. https://doi.org/10.24432/C53C8N

Download the zip, unzip it, and upload one of the three CSV files inside (e.g.
`s41598-020-73558-3_sepsis_survival_primary_cohort.csv`) using the uploader in this tab.
If you don't upload anything, the tab runs on a small **synthetic** dataset shaped like
the real one, clearly labelled, so you can see how everything works before downloading.

**What it shows:** survival rate broken down by age, sex, and number of prior sepsis
episodes, plus a simple logistic regression model trained on whichever data is loaded,
with its test-set accuracy and an interactive "try it" section.

**What it does not do, on purpose:** there is no drug name, dose, or treatment field
anywhere in this tab. Real public sepsis datasets don't include that information — it's
protected clinical data — and this app doesn't invent it either. The model's output is a
statistical pattern in one historical dataset, not a diagnosis or a real risk score for
any individual, and it is never connected to the fictional concentration simulator in the
other tabs.

The sidebar has three one-click scenarios: balanced comparison, accumulation, and
under-range. Useful when demonstrating to a judge who only has two minutes.

---

## The maths in plain English

The model treats the body as **one container** holding a fixed simulated volume of
fluid. Nothing about that is anatomically real; it is the simplest pharmacokinetic idea
that exists, chosen so the behaviour is easy to follow.

**Doses.** At each dosing time, the concentration in the container jumps upward by

```
jump = dose ÷ simulation volume
```

**Between doses.** The concentration falls away exponentially — fast at first, then
more slowly, the shape of a hot drink cooling:

```
C(t + Δt) = C(t) × e^(−k · Δt)
```

**The elimination rate `k`.** This is what controls how fast the fall is:

```
k = clearance ÷ simulation volume
```

A larger clearance gives a larger `k`, so the concentration drops faster between doses.

**The circadian wave.** In the fixed model, clearance is one number that never changes.
In the circadian-aware model it follows a smooth cosine wave across the day:

```
clearance(clock) = baseline × (1 + amplitude × cos(2π(clock − peak) ÷ 24))
```

`amplitude` comes from the circadian-strength setting (0%, 10%, 25% or 40%), and `peak`
is placed exactly 12 hours from the middle of your night window. So the **lowest**
clearance always lands in the middle of the night. With the default 23:00–07:00 window,
clearance peaks at 15:00 and bottoms out at 03:00.

**Why this changes the picture.** Lower clearance overnight means slower decay, so the
concentration falls less between night-time doses. The next dose then stacks on top of a
higher starting point, and the curve climbs. That is why the circadian line can drift
above the fixed line overnight — arithmetic, not a biological discovery.

**Stepping through time.** The simulation advances in steps of 0.02 hours (about 72
seconds), recalculating the local rate at each step and recording the concentration.
Over 48 hours that is roughly 2,400 data points, which is what makes the curve smooth.

**Measuring the result.** Area under the curve (AUC) is computed with the trapezoidal
rule — the total simulated exposure. Time above and below the thresholds is counted by
adding up the step length for every point outside the range. An alert event is counted
each time the curve *crosses* from inside the range to outside, so a single long
excursion counts once rather than thousands of times.

---

## Screenshots for a display board

Six images cover the project well. Take them at a wide browser window so the chart is
legible when printed.

1. **The header with the warning banner.** Your first panel. It establishes immediately
   that this is a simulation, which is the single most important thing a judge should
   read first.
2. **The main concentration chart, default settings.** Both lines visible, safety band
   shaded, night hours shaded, dose markers on. This is your hero image — print it
   largest.
3. **The same chart with circadian strength set to Strong.** Put it directly beside
   image 2. Side by side, the widening gap between the two lines is the whole result of
   the project in one glance.
4. **The comparison table plus the six metric cards.** Your numerical evidence. Crop
   tightly and print large enough to read from a metre away.
5. **The Emergency Override tab, unstable mode ON.** Shows the red status card and the
   fixed-parameter table. This is your responsible-design panel.
6. **The clearance wave chart.** Shows the input that causes everything else. Pair it
   with a one-sentence caption explaining that the trough sits in the middle of the
   night.

**Optional seventh:** a line of your `app.py` source showing the `clearance_at`
function, printed in a monospace font. Judges like seeing that you can point at the
equation in your own code.

**Caption every single image** with "Educational simulation — not for clinical use."
Consistency here is itself part of the project.

---

## Project structure

`app.py` is one file, divided into four commented sections:

1. **Configuration** — every fictional constant collected in one block, so a reader can
   see at a glance exactly which numbers are invented.
2. **Model** — pure mathematical functions with no interface code. These can be imported
   and tested on their own.
3. **Metrics** — the summary calculations and the interpretation text generator.
4. **User interface** — styling and the six tabs.

If you would rather split it up, move sections 2 and 3 into `model.py` and add
`from model import *` at the top of `app.py`. The functions have no Streamlit
dependencies, so they will move without changes.

---

## Troubleshooting

**`streamlit: command not found`** — the virtual environment is not active, or the
install did not finish. Re-run the activate command, then `pip install -r requirements.txt`.

**VS Code underlines the imports in red** — the editor is pointing at a different Python.
Press `Ctrl+Shift+P`, choose *Python: Select Interpreter*, and pick the one inside
`.venv`.

**The page loads but is blank** — refresh the browser. Streamlit occasionally needs one
reload after the first start.

**Port already in use** — run `streamlit run app.py --server.port 8502`.

---

## Safety statement

This project was built with deliberate limits. Doses are in invented simulation units
and never in milligrams. No real medicine is named or modelled. The safety thresholds
are fictional comparison markers, not clinical targets. Some antibiotics may carry
kidney-related toxicity risks, particularly if they accumulate — that is context for why
concentration over time is worth studying, and it is not a claim about anything this
program does. The emergency override deliberately makes the model *simpler and more
predictable* rather than more aggressive, because a model that is uncertain about its
assumptions should fall back, not push forward.
