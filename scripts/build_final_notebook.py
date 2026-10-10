"""Generate a new final notebook, without overwriting any existing notebook."""
import argparse
from pathlib import Path
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/FINAL_NOTEBOOK.ipynb"


def build():
    cells = []

    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def step(before, source, after):
        md(before)
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(source).strip()))
        md(after)

    md('''
    # US Accidents: learning to predict traffic impact

    **Group:** Yanis Barbieux, Kamar Benbacha, Théophile Melquiot

    **Hands-on Machine Learning, M1 Computer Science**

    ### AI use disclosure

    AI tools were used to help write and revise the documentation and Markdown,
    format the notebook, define and clarify variables, and write code.

    ## 1. Introduction

    Can the context of an accident help us identify a high impact on traffic?
    Our target is **Severity 3 or 4 versus Severity 1 or 2**. It describes traffic
    disruption, rather than injuries or the risk that an accident will occur.

    We developed separate approaches and brought them together. Kamar explored
    the columns and missing values, prepared a larger sample, and compared
    logistic regression with two tree models. Yanis focused on time-based
    evaluation, repeated events and boosting with native categorical features.
    Their differences gave us concrete questions: which records should we keep,
    how should we encode categories, and does a larger sample help?

    This notebook develops the common approach step by step. We use Source2,
    compare boosting with a random forest, and explain the choices supported by
    validation. Weather is examined both at accident start and in an update
    allowing an observation up to one hour later.

    Dataset: [US Accidents, March 2023 version](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
    Severity definition: [creator metadata](https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata).
    ''')
    md('''
    ### Related work and our contribution

    Moosavi et al., [A Countrywide Traffic Accident Dataset (2019)](https://arxiv.org/abs/1906.05409),
    introduced US-Accidents with time, location, weather and nearby points of
    interest. Its original release is smaller than the March 2023 file we use.
    Their [Accident Risk Prediction based on Heterogeneous Sparse Data (2019)](https://arxiv.org/abs/1909.09638)
    describes a deep learning approach to predicting accident occurrence, which
    differs from predicting the traffic impact of an already recorded accident.
    These points come from the abstracts; we do not compare their scores with ours.

    Our contribution is the combination of a clear preparation, controlled
    comparisons and an evaluation across years. We use existing algorithms and
    focus on understanding what the data and the results allow us to conclude.
    ''')
    step('''
    ## 2. Preprocessing

    ### 2.1 Set up one common configuration

    We keep the main choices together so the study is easy to reproduce. The CSV
    is kept locally and is not part of the submission. Install
    `requirements-final-notebook.txt` and open the notebook within the project.
    Python modules handle reusable data access; the main transformations and
    model training remain visible below.
    ''', '''
    import os
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[name] = "4"
    from pathlib import Path
    import sys, json, itertools, gc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown, HTML
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (average_precision_score, roc_auc_score,
        precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix)
    from threadpoolctl import threadpool_limits

    ROOT = Path.cwd().resolve()
    if (ROOT / "us_accidents_traffic_impact_prediction/traffic_impact").is_dir():
        ROOT = ROOT / "us_accidents_traffic_impact_prediction"
    while not (ROOT / "traffic_impact").is_dir() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    assert (ROOT / "traffic_impact").is_dir(), "Open this notebook inside the project."
    sys.path.insert(0, str(ROOT))
    from traffic_impact.final_notebook import find_csv, load_samples, data_dictionary, nearby_keys
    from traffic_impact.experiment import POI, LIGHT, WNUM, WCAT, FORBIDDEN, preprocessor
    CSV_PATH = find_csv()  # Change this path if the CSV is stored elsewhere.
    WORK = ROOT / "data/final_notebook"
    REPORTS, FIGURES = WORK / "reports", WORK / "figures"
    for path in [REPORTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
    SEED, TRAIN_N, WEATHER_AFTER, WEATHER_CAP = 42, 500_000, 60, 25
    COMPARISONS = ROOT / "results/final_notebook_comparison"
    pd.set_option("display.max_columns", 9)
    plt.rcParams.update({"figure.figsize": (8, 3.6), "figure.dpi": 100})
    display(pd.Series({"Training events": TRAIN_N, "Weather update (minutes)": WEATHER_AFTER,
        "Weather category limit": WEATHER_CAP, "Seed": SEED}, name="Common choices").to_frame())
    ''', '''
    The configuration uses a fixed seed and keeps the natural class proportions.
    Time and location always refer to the accident start; only the weather window
    changes between the two scenarios.
    ''')
    step('''
    ### 2.2 Look at the original records

    Kamar began by reading a small part of the file before loading larger samples.
    We keep this useful first step: identify the columns and their units, then
    inspect a few records. The first rows illustrate the format; they are not a
    representative sample for drawing conclusions.
    ''', '''
    preview = pd.read_csv(CSV_PATH, nrows=5)
    dictionary = data_dictionary(preview.columns)
    dictionary.to_csv(REPORTS / "data_dictionary.csv", index=False)
    display(preview[["Source", "Severity", "Start_Time", "State", "Temperature(F)",
                     "Weather_Timestamp"]])
    input_groups = pd.DataFrame({
        "Group": ["Time", "Location", "Road context", "Light", "Weather"],
        "Inputs": [4, 4, len(POI), len(LIGHT), len(WNUM) + len(WCAT)],
        "Examples": ["hour, weekday, month, weekend", "latitude, longitude, State, Timezone",
                     "junction, crossing, signals", "day/night and twilight",
                     "temperature, visibility, wind, conditions"]})
    display(input_groups)
    ''', '''
    The file mixes numerical measurements, categories, flags and timestamps.
    Units such as Fahrenheit, miles and inches are kept as recorded. The complete
    dictionary in the appendix defines every original column, including fields
    we exclude, and the four derived time variables.
    ''')
    step('''
    ### 2.3 Work with distinct events

    We read the full file in blocks when samples must be prepared. Uniform event
    priorities with seed 42 select distinct Source2 events within each period.
    Priorities do not depend on Severity. The 200,000-event sample will be a
    subset of the 500,000-event sample.

    An event key uses Source, start/end time, start coordinates, Distance and
    Description, without Severity. Repeated copies are grouped; events with
    conflicting Severity labels are excluded, even if their binary labels agree.
    Some identity fields are excluded from prediction. Nearby accidents are not
    automatically treated as the same event.
    ''', '''
    raw, audit, AUDIT_FOLDER = load_samples(CSV_PATH, WORK)
    raw["train"] = raw["train"].nsmallest(TRAIN_N, "_priority").sort_values("_t").reset_index(drop=True)
    audit_summary = pd.Series({"Original records": audit["rows"],
        "Original columns": audit["columns"], "Source2 records": audit["source2_rows"],
        "Repeated Source2 event records": audit["source2_duplicate_event_rows"],
        "Conflicting Source2 event keys": audit["source2_conflicting_event_keys"]}, name="Full-file audit")
    display(audit_summary.to_frame())
    ''', '''
    The full-file audit distinguishes repeated events from record identifiers.
    All copies of a conflicting event are excluded because the file does not
    tell us which label is correct. Sampling keeps the study manageable without
    forcing the two classes to have equal sizes. Verified caches are reused when
    the CSV, preparation code, environment and file hashes match.
    ''')
    step('''
    ### 2.4 Compare sources before choosing the study population

    Both individual approaches examined the providers. We plot the proportion
    of positive labels by source and year, using the full-file audit. This helps
    us decide whether mixing sources would make the target difficult to interpret.
    ''', '''
    source_month = pd.read_csv(AUDIT_FOLDER / "monthly_source_severity.csv")
    source_month["year"] = pd.to_numeric(source_month["_month"].str[:4], errors="coerce")
    source_month["positives"] = source_month["count"].where(source_month.Severity.ge(3), 0)
    source_year = source_month.groupby(["Source", "year"])[["count", "positives"]].sum()
    source_year["positive_rate"] = source_year.positives / source_year["count"]
    for source, group in source_year.reset_index().groupby("Source"):
        plt.plot(group.year, group.positive_rate, marker="o", label=source)
    plt.xlabel("Year")
    plt.ylabel("Share of Severity 3 or 4")
    plt.title("The label distribution differs between sources")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "01_sources.png")
    plt.show()
    ''', '''
    The sources have different label distributions and changes over time.
    We focus on Source2 to give the study a clear scope. This does not establish
    the cause of changes in other sources, or prove that Source2 is always better.
    Source2 ends in September 2022, so our last period is incomplete.
    ''')
    step('''
    ### 2.5 Clean invalid values without deleting incomplete weather records

    Kamar removed rows with more than four missing predictors, then filled the
    remaining gaps using training statistics. For the common study, we keep
    records without weather: their time, location and road context remain useful.
    This avoids selecting only accidents whose weather is better documented.

    A valid date and a label in 1-4 are required. Impossible coordinates, humidity
    or negative physical measurements become missing; plausible extremes remain.
    We do not impute the target. Temperature has no arbitrary cutoff.
    ''', '''
    raw_clean, cleaning_rows, invalid_rows = {}, [], []
    bounds = {"Start_Lat": (-90, 90), "Start_Lng": (-180, 180), "Humidity(%)": (0, 100),
        "Pressure(in)": (0, np.inf), "Visibility(mi)": (0, np.inf),
        "Wind_Speed(mph)": (0, np.inf), "Precipitation(in)": (0, np.inf)}
    for split, original in raw.items():
        frame = original.copy()
        frame["_t"] = pd.to_datetime(frame.Start_Time, format="mixed", errors="coerce")
        valid = frame._t.notna() & frame.Severity.isin([1, 2, 3, 4])
        cleaning_rows.append({"Period": split, "Before": len(frame),
            "Invalid date/label": int((~valid).sum()), "After": int(valid.sum())})
        frame = frame.loc[valid].reset_index(drop=True)
        for column, (lower, upper) in bounds.items():
            values = pd.to_numeric(frame[column], errors="coerce")
            bad = values.notna() & (~values.between(lower, upper) | ~np.isfinite(values))
            invalid_rows.append({"split": split, "column": column, "changed": int(bad.sum())})
            frame[column] = values.mask(bad)
        raw_clean[split] = frame
    cleaning = pd.DataFrame(cleaning_rows)
    invalid = pd.DataFrame(invalid_rows)
    display(cleaning)
    display(invalid.groupby("column").changed.sum().rename("Invalid values replaced").to_frame())
    cleaning.to_csv(REPORTS / "record_cleaning.csv", index=False)
    invalid.to_csv(REPORTS / "numeric_cleaning.csv", index=False)
    ''', '''
    These samples contain no invalid dates, labels or values under the specified
    physical bounds, so this cleaning step keeps their records unchanged. This
    does not guarantee that the entire file is error-free. Keeping incomplete weather is a
    choice about the population studied, not a measured victory over Kamar's
    row-removal policy.
    ''')
    step('''
    ### 2.6 Keep the evaluation in chronological order

    Kamar used full evaluation years with boundaries in September; Yanis used
    calendar years. The common study keeps calendar years for clarity. We learn
    on 2016-2019, compare settings on 2020, and examine 2021 and the available
    part of 2022 afterwards. Labels remain separate from predictors.
    ''', '''
    period_names = {"train": "2016-2019 (train)", "validation": "2020 (validation)",
        "test": "2021 (historical)", "stress": "2022 to Sep 8"}
    y = {s: d.Severity.ge(3).to_numpy(dtype="int8") for s, d in raw_clean.items()}
    sample_table = pd.DataFrame([{"Period": period_names[s], "Events": len(d),
        "Positive share (%)": round(100 * y[s].mean(), 1),
        "First date": str(d._t.min().date()), "Last date": str(d._t.max().date())}
        for s, d in raw_clean.items()])
    display(sample_table)
    levels = pd.DataFrame({period_names[s]: d.Severity.value_counts().reindex([1, 2, 3, 4], fill_value=0)
        for s, d in raw_clean.items()}).T
    levels.div(levels.sum(axis=1), axis=0).plot.bar(stacked=True, figsize=(8, 3.6))
    plt.ylabel("Share of events")
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.legend(title="Severity", bbox_to_anchor=(1.02, 1))
    plt.title("Binary labels hide differences between the original levels")
    plt.tight_layout()
    plt.savefig(FIGURES / "02_levels.png")
    plt.show()
    ''', '''
    Around one third of the sampled accidents belong to the positive class, so
    accuracy alone would be insufficient. Most positive records are Severity 3;
    Severity 4 is rare. The share of Severity 1 also changes in 2022. Later-year
    results must be read with these population differences in mind. The project
    has already examined 2021 and 2022; these are historical evaluations, not new
    independent tests.
    ''')
    step('''
    ### 2.7 Derive features that describe the accident context

    Following the individual notebooks, we turn Start_Time into hour, weekday,
    month and a weekend flag. We keep start location, nearby road features and
    the light indicators. End time, end coordinates, Distance and Description
    are excluded because their initial availability is not established.
    ''', '''
    def initial_features(frame):
        time = frame["_t"]
        result = pd.DataFrame({"hour": time.dt.hour, "day_of_week": time.dt.dayofweek,
            "month": time.dt.month, "weekend": time.dt.dayofweek.ge(5).astype(int)})
        for column in ["Start_Lat", "Start_Lng", "State", "Timezone"] + POI + LIGHT + WNUM + WCAT:
            result[column] = frame[column]
        for column in POI:
            result[column] = result[column].astype(float)
        for column in ["State", "Timezone"] + LIGHT + WCAT:
            result[column] = result[column].astype("string")
        return result

    initial = {s: initial_features(d) for s, d in raw_clean.items()}
    display(initial["train"][["hour", "day_of_week", "month", "weekend"]].head())
    by_hour = pd.Series(y["train"]).groupby(initial["train"].hour).mean()
    by_hour.plot(marker="o")
    plt.xlabel("Local accident start hour")
    plt.ylabel("Share of high-impact labels")
    plt.title("Time patterns in the training sample")
    plt.tight_layout()
    plt.savefig(FIGURES / "03_hour.png")
    plt.show()
    ''', '''
    The curve describes an association in the training data, not a causal effect
    of the hour. We do not include the year as a predictor. Turning_Loop is kept
    in the 34-input definition, but it is always False in the audited file, so it
    provides no learned signal here. Day/night indicators describe natural light
    at the start, not artificial road lighting.
    ''')
    step('''
    ### 2.8 Clarify the two weather scenarios with an example

    The file contains one weather observation already associated with each
    accident. We check its timestamp; we cannot retrieve a different reading.
    Kamar accepted observations within two hours before or after the start.
    Yanis initially accepted only the two preceding hours. We compare that strict
    rule with an update allowing up to one hour after the start.
    ''', '''
    example = pd.DataFrame({"Accident start": ["10:00"] * 5,
        "Associated observation": ["07:59", "08:00", "09:30", "10:30", "11:01"],
        "Offset (minutes)": [-121, -120, -30, 30, 61]})
    example["At start"] = example["Offset (minutes)"].between(-120, 0)
    example["With update"] = example["Offset (minutes)"].between(-120, 60)
    display(example)
    ''', '''
    This is an illustrative example, not five sampled accidents. A 10:30
    observation cannot be used for a prediction at 10:00. The update is a later
    prediction; even an eligible timestamp does not establish when the reading
    was published. Other future fields do not receive this weather exception.
    ''')
    step('''
    ### 2.9 Apply the weather rule and harmonise wind labels

    All nine weather fields are masked together if the timestamp is missing or
    outside the chosen window. The accident remains. We also merge spelling
    variants such as East/E, without merging different directions such as NE/NW.
    ''', '''
    aliases = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
        "CALM": "Calm", "VAR": "Variable", "VARIABLE": "Variable"}
    def weather_features(frame, after=60):
        result = initial_features(frame)
        observed = pd.to_datetime(frame.Weather_Timestamp, format="mixed", errors="coerce")
        offset = (observed - frame["_t"]).dt.total_seconds() / 60
        eligible = offset.between(-120, after)
        for column in WNUM:
            result[column] = pd.to_numeric(result[column], errors="coerce").where(eligible)
        for column in WCAT:
            values = result[column]
            if column == "Wind_Direction":
                values = values.str.strip().str.upper().replace(aliases)
            result[column] = values.where(eligible)
        return result, eligible

    features, weather_rows = {}, []
    for split, frame in raw_clean.items():
        features[split], _ = weather_features(frame, WEATHER_AFTER)
        for after, scenario in [(0, "At start"), (WEATHER_AFTER, "With update")]:
            _, eligible = weather_features(frame, after)
            weather_rows.append({"Period": period_names[split], "Scenario": scenario,
                "Eligible timestamps (%)": round(100 * eligible.mean(), 1)})
    weather_table = pd.DataFrame(weather_rows)
    display(weather_table[weather_table.Period.eq(period_names["validation"])])
    weather_table.to_csv(REPORTS / "weather_coverage.csv", index=False)
    ''', '''
    The wider window keeps substantially more associated observations. Timestamp
    eligibility is not the same as complete weather: individual fields can still
    be missing. We retain the update as the main scenario and keep the strict rule
    as a comparison, with its different prediction time clearly stated.
    ''')
    step('''
    ### 2.10 Inspect missing values before deciding how to handle them

    Kamar's missingness analysis showed why removing every incomplete row would
    discard too much data. Here we compare the same training events before and
    after timestamp masking. The masking can increase missingness without deleting
    records. Boosting can retain numerical gaps; the forest will receive medians.
    ''', '''
    missingness = pd.DataFrame({"Before timestamp rule (%)": initial["train"].isna().mean() * 100,
        "After timestamp rule (%)": features["train"].isna().mean() * 100})
    missingness = missingness.loc[WNUM + WCAT].round(1)
    display(missingness)
    missingness.plot.barh(figsize=(8, 4))
    plt.xlabel("Missing values (%)")
    plt.ylabel("")
    plt.title("Weather gaps are retained explicitly")
    plt.tight_layout()
    plt.savefig(FIGURES / "04_missingness.png")
    plt.show()
    missingness.to_csv(REPORTS / "missingness_before_after.csv")
    ''', '''
    Precipitation and wind chill have the most missing values in the training
    sample, even before timestamp masking. Missing weather does not erase the other
    context. Each model uses an appropriate representation of absence. This does
    not show that missingness itself causes higher or lower traffic impact.
    ''')
    step('''
    ### 2.11 Learn categorical vocabularies on training data only

    We keep Kamar's idea of grouping rare weather conditions into Other. A category
    needs at least 100 training occurrences; Weather_Condition keeps at most the
    25 most frequent eligible labels. Missing is separate from Other. Later data
    cannot change the vocabulary. Numerical and road inputs use compact types.
    ''', '''
    class CategoryPreparation:
        def __init__(self, weather_cap=25):
            self.weather_cap = weather_cap

        def fit(self, frame):
            self.columns = [c for c in frame if c == "Turning_Loop" or frame[c].nunique(dropna=False) > 1]
            self.vocab = {}
            for column in self.columns:
                if pd.api.types.is_string_dtype(frame[column].dtype):
                    counts = frame[column].value_counts()
                    keep = counts[counts >= 100].index.tolist()
                    if column == "Weather_Condition":
                        keep = keep[:self.weather_cap]
                    self.vocab[column] = [str(v) for v in keep if v not in ["Missing", "Other"]]
            return self

        def transform(self, frame):
            result = frame[self.columns].copy()
            for column, vocabulary in self.vocab.items():
                values = result[column].fillna("Missing").astype(str)
                values = values.where(values.isin(vocabulary + ["Missing"]), "Other")
                result[column] = pd.Categorical(values, categories=vocabulary + ["Missing", "Other"])
            for column in self.columns:
                if column not in self.vocab:
                    result[column] = result[column].astype("float32")
            return result

    category_prep = CategoryPreparation(WEATHER_CAP).fit(features["train"])
    X = {s: category_prep.transform(frame) for s, frame in features.items()}
    display(pd.DataFrame([{"Feature": c, "Retained training labels": len(v),
        "Missing label": "Missing", "Rare/unseen label": "Other"} for c, v in category_prep.vocab.items()]))
    ''', '''
    The vocabulary has been learned once, then reused for all periods. The native
    categorical dtype tells boosting that categories have no numerical order.
    All 34 logical inputs are kept in this sample, including constant Turning_Loop.
    We have not demonstrated that each of them is necessary.
    ''')
    step('''
    ### 2.12 Make the transformations concrete

    We inspect a few real category transformations and a small one-hot example.
    One-hot creates an indicator for each category: it avoids inventing an order
    between states, but increases the number of columns. We adapt the preparation
    to each model rather than assuming one representation is always best.
    ''', '''
    transformed = pd.DataFrame({"Recorded wind": raw_clean["train"].Wind_Direction.astype("string"),
        "After weather/spelling rule": features["train"].Wind_Direction,
        "Final wind category": X["train"].Wind_Direction.astype("string")})
    display(transformed.drop_duplicates().head(8).fillna("<missing>"))
    display(pd.get_dummies(pd.DataFrame({"State": ["CA", "NY", "TX"]}), dtype=int))
    transformation_summary = pd.DataFrame([{"Step": "Start_Time to four time inputs", "Output": "hour, weekday, month, weekend"},
        {"Step": "Boolean road indicators", "Output": "0/1"},
        {"Step": "Weather outside window", "Output": "Missing weather; accident kept"},
        {"Step": "Rare or new categorical value", "Output": "Other"},
        {"Step": "Missing categorical value", "Output": "Missing"}])
    display(transformation_summary)
    ''', '''
    The wind table shows observed transformations. The three-state table is a
    teaching example. One-hot does not add new observations or fill weather gaps.
    We use native categories for boosting and one-hot for the forest. The study
    does not directly compare these two encodings with the same boosting model.
    ''')
    md('''
    ## 3. Training

    ### From Kamar's first models to the common comparison

    Kamar trained a constant baseline, logistic regression, boosting and a forest
    on her prepared data. Logistic regression improved on the baseline; both tree
    models performed better on her validation, with boosting ahead of the forest.
    This gave us a useful reason to examine trees more closely.

    We do not directly rank Kamar's and Yanis's notebooks: their periods, samples
    and preprocessing differ. Below, boosting and the forest are trained again
    on a shared protocol. Earlier common validation comparisons explain the
    settings; later years do not select them.
    ''')
    step('''
    ### 3.1 Check model capacity, then training volume

    A larger leaf limit lets each boosting tree distinguish more combinations of
    conditions. More iterations add correction steps, but both choices can lead
    to fitting training details too closely. We examined a small set of settings
    on 2020. After selecting the tree settings, we checked the two sample sizes
    again under matching conditions. These tables summarise saved validation
    comparisons, rather than launching a new parameter search in this notebook.
    ''', '''
    validation_trials = pd.read_csv(COMPARISONS / "validation.csv")
    capacity_names = ["e07_hgb_500000_w60_c15", "e10_hgb_500000_w60_c15", "e12_hgb_500000_w60_c15"]
    capacity = validation_trials.set_index("name").loc[capacity_names].copy()
    capacity["Maximum leaves"] = capacity.params.map(lambda p: json.loads(p)["max_leaf_nodes"])
    capacity["Iterations"] = capacity.params.map(lambda p: json.loads(p)["max_iter"])
    display(capacity[["Maximum leaves", "Iterations", "ap"]].rename(columns={"ap": "Validation AP"}).reset_index(drop=True).round(3))
    volume_names = ["e19_hgb_200000_w60_c25_refined", "e21_hgb_500000_w60_c25_refined"]
    volume = validation_trials.set_index("name").loc[volume_names, ["train_n", "ap"]]
    display(volume.rename(columns={"train_n": "Training events", "ap": "Validation AP"}).reset_index(drop=True).round(3))
    ''', '''
    The capacity comparisons support 63 leaves per tree and 400 iterations. With
    these settings, the larger sample also improves validation AP, so we retain
    500,000 events. We learn from changing one aspect while keeping the comparison
    conditions clear; the results do not establish a universal optimum.
    ''')
    step('''
    ### 3.2 Refine the weather grouping and check the prediction-time tradeoff

    The number of frequent Weather_Condition labels is a separate preparation
    choice. We also compare the strict and updated weather scenarios at the
    reference model settings. Each question has a small table, with one metric
    and no execution log.
    ''', '''
    category_names = ["e20_hgb_500000_w60_c10_refined", "e12_hgb_500000_w60_c15", "e21_hgb_500000_w60_c25_refined"]
    categories = validation_trials.set_index("name").loc[category_names, ["cap", "ap"]]
    display(categories.rename(columns={"cap": "Weather category limit", "ap": "Validation AP"}).reset_index(drop=True).round(3))
    weather_comparison = validation_trials.set_index("name").loc[
        ["e05_hgb_500000_w0_c15", "e07_hgb_500000_w60_c15"], ["after", "ap"]].copy()
    weather_comparison.index = ["At start", "With update"]
    display(weather_comparison.rename(columns={"after": "Minutes after start allowed", "ap": "Validation AP"}).round(3))
    ''', '''
    The limit of 25 has the highest AP among the three grouping limits, but the
    differences are small: this is a modest refinement. The weather update gives
    a modest improvement at the reference settings and much greater timestamp
    coverage. This gain was not measured again with the final tuned settings.
    The update also requires a different prediction time from the strict scenario.
    ''')
    step('''
    ### 3.3 Fit the main boosting model

    Boosting builds trees sequentially using the gradient of the loss. Numerical
    bins make its split search efficient; categorical inputs are handled natively.
    We now train the selected configuration on 2016-2019. There is no class
    weighting, synthetic balancing or random early-stopping split.
    ''', '''
    boosting_settings = {"max_iter": 400, "max_leaf_nodes": 63, "learning_rate": 0.1,
        "categorical_features": "from_dtype", "early_stopping": False, "random_state": SEED}
    boosting = HistGradientBoostingClassifier(**boosting_settings)
    with threadpool_limits(limits=4):
        boosting.fit(X["train"], y["train"])
    prior = float(y["train"].mean())
    display(pd.Series({"Training events": len(y["train"]), "Logical inputs": X["train"].shape[1],
        "Maximum leaves per tree": boosting.max_leaf_nodes, "Iterations": boosting.n_iter_}, name="Boosting").to_frame())
    ''', '''
    The model uses the larger sample and the selected tree capacity. The constant
    baseline assigns the training positive share to every accident; it will help
    us judge how much the context improves ranking.
    ''')
    step('''
    ### 3.4 Fit a forest with its own preparation

    A random forest averages trees fitted on bootstrap samples, with random
    feature subsets at each split. We keep 100 trees and at least five events per
    leaf, as selected on the common validation. Numerical gaps receive training
    medians and categories receive one-hot indicators. The selected forest uses
    a weather limit of 15, while boosting uses 25; each keeps its validated setup.
    ''', '''
    forest_categories = CategoryPreparation(weather_cap=15).fit(features["train"])
    forest_train = forest_categories.transform(features["train"])
    forest_pre = preprocessor(forest_train)
    forest_matrix = forest_pre.fit_transform(forest_train).astype("float32").toarray()
    forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
        max_features="sqrt", random_state=SEED, n_jobs=4)
    with threadpool_limits(limits=4):
        forest.fit(forest_matrix, y["train"])
    display(pd.Series({"Training events": len(y["train"]), "Logical inputs": forest_train.shape[1],
        "Encoded columns": forest_matrix.shape[1], "Trees": forest.n_estimators}, name="Forest").to_frame())
    del forest_matrix, forest_train
    _ = gc.collect()
    ''', '''
    The forest is a real trained competitor. Its encoded matrix has more columns
    than the 34 logical inputs because one variable can have several indicators.
    Medians and encoding rules were fitted on training data only. This comparison
    assesses the selected pipelines, not the algorithm alone at identical encodings.
    ''')
    step('''
    ### 3.5 Compare validation predictions and fix decision thresholds

    AP summarises precision across recall levels and measures ranking quality.
    ROC-AUC measures positive-versus-negative ranking. Neither is classification
    accuracy. For binary decisions, we choose the threshold that maximises F1 on
    validation, balancing precision and recall, then keep it fixed for later years.
    ''', '''
    def evaluate(target, prediction, threshold, priorities):
        positive = prediction >= threshold
        budget = max(1, int(np.ceil(0.10 * len(target))))
        chosen = np.lexsort((priorities, -prediction))[:budget]
        return {"Events": len(target), "Positive share": float(target.mean()),
            "AP": average_precision_score(target, prediction), "ROC-AUC": roc_auc_score(target, prediction),
            "Precision": precision_score(target, positive, zero_division=0),
            "Recall": recall_score(target, positive, zero_division=0),
            "F1": f1_score(target, positive, zero_division=0), "Threshold": float(threshold),
            "Precision top 10%": float(target[chosen].mean()),
            "Recall top 10%": float(target[chosen].sum() / target.sum())}

    def f1_threshold(target, prediction):
        precision, recall, thresholds = precision_recall_curve(target, prediction)
        f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
            out=np.zeros_like(thresholds), where=precision[:-1] + recall[:-1] > 0)
        return float(thresholds[np.argmax(f1)])

    predictions = {s: {"Constant": np.full(len(y[s]), prior)} for s in y}
    def predict_forest(split):
        encoded = forest_pre.transform(forest_categories.transform(features[split]))
        return forest.predict_proba(encoded.astype("float32").toarray())[:, 1]
    with threadpool_limits(limits=4):
        predictions["validation"]["Boosting"] = boosting.predict_proba(X["validation"])[:, 1]
        predictions["validation"]["Forest"] = predict_forest("validation")
    thresholds = {name: f1_threshold(y["validation"], values)
        for name, values in predictions["validation"].items()}
    score_rows = [{"Period": "validation", "Model": name,
        **evaluate(y["validation"], values, thresholds[name], raw_clean["validation"]._priority.to_numpy())}
        for name, values in predictions["validation"].items()]
    validation_scores = pd.DataFrame(score_rows)
    display(validation_scores[["Model", "AP", "ROC-AUC", "Precision", "Recall", "F1"]].round(3))
    ''', '''
    The common validation comparison supports boosting as the main model among
    the settings examined. The forest also improves on the constant baseline.
    Validation F1 benefits from selecting the threshold on this same data, so it
    is not an independent estimate. Parameters, vocabularies and thresholds are
    now fixed before the later-period evaluation.
    ''')
    step('''
    ## 4. Evaluation

    ### 4.1 Examine later periods with the fixed models

    We apply the fitted models and unchanged preparations to 2021 and the
    available part of 2022. We keep the table compact and compare AP with the
    prevalence reference, since the positive share changes between years.
    ''', '''
    with threadpool_limits(limits=4):
        for split in ["test", "stress"]:
            predictions[split]["Boosting"] = boosting.predict_proba(X[split])[:, 1]
            predictions[split]["Forest"] = predict_forest(split)
            for name, values in predictions[split].items():
                score_rows.append({"Period": split, "Model": name,
                    **evaluate(y[split], values, thresholds[name], raw_clean[split]._priority.to_numpy())})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(REPORTS / "model_scores.csv", index=False)
    compact_scores = scores[scores.Period.isin(["test", "stress"])][
        ["Period", "Model", "AP", "ROC-AUC", "Precision", "Recall", "F1"]].copy()
    compact_scores["Period"] = compact_scores.Period.map(period_names)
    display(compact_scores.round(3))
    ''', '''
    Boosting remains ahead of the selected forest on these historical periods.
    Similar scores across years are encouraging for this scope, but changes in
    labels and incomplete coverage limit the interpretation. These years had
    already been explored in the project; a new confirmation would require a
    later independent period.
    ''')
    step('''
    ### 4.2 Visualise precision, recall and temporal ranking

    The precision-recall curve shows the tradeoff across thresholds. Its reference
    line is the positive share of the evaluated sample. A second figure compares
    AP across years, with the baseline included to show changing prevalence.
    ''', '''
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in ["Boosting", "Forest"]:
        precision, recall, _ = precision_recall_curve(y["test"], predictions["test"][name])
        axes[0].plot(recall, precision, label=name)
    axes[0].axhline(y["test"].mean(), color="gray", linestyle="--", label="Positive share")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Historical 2021 evaluation", ylim=(0, 1.02))
    axes[0].legend()
    ranking = scores.pivot(index="Period", columns="Model", values="AP").reindex(["validation", "test", "stress"])
    ranking.index = ["2020", "2021", "2022 partial"]
    ranking[["Constant", "Forest", "Boosting"]].plot.bar(ax=axes[1], rot=0)
    axes[1].set(ylabel="Average precision", xlabel="", title="Ranking across periods", ylim=(0, 1))
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "05_evaluation.png")
    plt.show()
    ''', '''
    Better ranking does not remove the precision-recall tradeoff. A more permissive
    threshold finds more positives but also creates more false alerts. AP and
    ROC-AUC describe ranking; they do not show that the numerical scores are
    calibrated probabilities.
    ''')
    step('''
    ### 4.3 Understand the remaining mistakes

    We inspect the 2021 confusion matrix at the fixed F1 threshold and check
    recall separately for Severity 3 and 4. This avoids letting the common level 3
    hide the rare level 4. These diagnostics do not select a new model or threshold.
    ''', '''
    positive_test = predictions["test"]["Boosting"] >= thresholds["Boosting"]
    matrix = confusion_matrix(y["test"], positive_test, labels=[0, 1])
    display(pd.DataFrame(matrix, index=["Actual low impact", "Actual high impact"],
        columns=["Predicted low impact", "Predicted high impact"]))
    severity_diagnostics = []
    for level in [3, 4]:
        mask = raw_clean["test"].Severity.eq(level).to_numpy()
        severity_diagnostics.append({"Severity": level, "Events": int(mask.sum()),
            "Recall (%)": round(100 * positive_test[mask].mean(), 1)})
    display(pd.DataFrame(severity_diagnostics))
    test_boosting = scores[(scores.Period == "test") & (scores.Model == "Boosting")].iloc[0]
    display(Markdown(f"At the fixed threshold, boosting finds about **{100*test_boosting.Recall:.0f}%** "
        f"of high-impact events. About **{100*test_boosting.Precision:.0f}%** of its positive predictions "
        "are actually high-impact records. Both missed events and false alerts remain."))
    ''', '''
    The overall binary metrics are mostly driven by Severity 3. The smaller count
    for level 4 makes its recall less secure as a general conclusion. The matrix
    describes errors under our study labels, not injury outcomes or operational
    intervention costs.
    ''')
    step('''
    ### 4.4 Look for patterns in errors without changing the model

    As a first descriptive analysis, we group 2021 by start-hour range. We show
    both missed-positive and false-positive rates with their denominators. A
    difference can reflect population composition; it is not evidence of causation.
    ''', '''
    errors = pd.DataFrame({"Hour group": pd.cut(raw_clean["test"]._t.dt.hour,
        bins=[-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]),
        "Actual": y["test"], "Predicted": positive_test})
    error_rows = []
    for group, frame in errors.groupby("Hour group", observed=True):
        actual = frame.Actual.eq(1)
        negatives = ~actual
        error_rows.append({"Start hour": str(group), "Events": len(frame),
            "Positives": int(actual.sum()), "Negatives": int(negatives.sum()),
            "Missed positives (%)": 100 * (~frame.loc[actual, "Predicted"]).mean(),
            "False positives (%)": 100 * frame.loc[negatives, "Predicted"].mean()})
    error_table = pd.DataFrame(error_rows)
    display(error_table.round(1))
    error_table.to_csv(REPORTS / "errors_by_hour.csv", index=False)
    ''', '''
    In this sample, the missed-positive rate is higher in the morning than in the
    evening. This gives us a concrete subgroup to investigate further. The two
    error types have different denominators, and population differences can
    contribute to the pattern. We keep the model fixed rather than tuning it on
    this historical evaluation.
    ''')
    step('''
    ### 4.5 Distinguish a classification threshold from an alert budget

    Selecting only the highest 10% of scores is a different decision from the
    F1 threshold. We report the main model under both policies. Ties in the budget
    are broken using event priorities independent of the labels. The budget is
    retrospective, not a validated streaming alert service.
    ''', '''
    policies = pd.DataFrame([
        {"Policy": "Validation-selected F1 threshold", "Precision (%)": 100 * test_boosting.Precision,
         "Recall (%)": 100 * test_boosting.Recall},
        {"Policy": "Highest 10% of scores", "Precision (%)": 100 * test_boosting["Precision top 10%"],
         "Recall (%)": 100 * test_boosting["Recall top 10%"]}])
    display(policies.round(1))
    ''', '''
    Concentrating on fewer events can raise precision while finding a smaller
    share of all positives. These recalls should never be presented as the same
    measure. A deployment policy would also need costs, publication times and
    probability calibration, which are outside this study.
    ''')
    step('''
    ## 5. Conclusion

    We bring the main observations together, keeping the numerical summary short.
    The following paragraph uses the scores calculated in this execution.
    ''', '''
    partial_boosting = scores[(scores.Period == "stress") & (scores.Model == "Boosting")].iloc[0]
    display(Markdown(f"Our selected boosting model reaches **AP {test_boosting.AP:.3f}** "
        f"in the historical 2021 evaluation and **{partial_boosting.AP:.3f}** in the available part of 2022. "
        "It improves on the selected forest and the constant reference under the common protocol. "
        "The main choices are a larger training sample and a more capable boosting configuration; "
        "the weather category limit is a smaller refinement."))
    ''', '''
    Combining the approaches helped us understand that preprocessing and evaluation
    belong together. Kamar's exploration and first models made the preparation
    choices concrete. Yanis's event and temporal checks made the comparisons
    easier to interpret. A higher score is meaningful only when we know its data,
    permitted information and prediction time.

    The model still makes mistakes. Results concern one provider, most positives
    are level 3, and 2022 is incomplete. Weather observation times do not establish
    publication times, and the update is a later prediction. We have not calibrated
    the scores as probabilities or confirmed the model on a newly unseen period.
    With more time, we would extend error analysis to locations, study calibration,
    and evaluate on a later independent period.
    ''')
    step('''
    ## Appendix A. Full data dictionary

    Each original column and each derived input is defined below. Expand a field
    to see its type, unit, role, time reference and treatment. The same dictionary
    is exported to `data/final_notebook/reports/data_dictionary.csv`. The expandable
    entries use lightweight HTML, without an interactive plotting dependency.
    ''', '''
    from html import escape
    entries = []
    for row in dictionary.to_dict("records"):
        body = "".join(f"<dt><b>{escape(key)}</b></dt><dd>{escape(str(value))}</dd>"
            for key, value in row.items() if key not in ["Column", "Meaning"])
        entries.append(f"<details style='margin:6px 0'><summary><b>{escape(row['Column'])}</b>: "
            f"{escape(row['Meaning'])}</summary><dl>{body}</dl></details>")
    display(HTML("".join(entries)))
    ''', '''
    The dictionary distinguishes excluded columns from fields used only for an
    audit. The data's measurement units are kept explicit. The availability
    descriptions are study assumptions where publication times are absent.
    ''')
    step('''
    ## Appendix B. Reproduction and consistency

    We check periods, event uniqueness, input exclusions and fixed category rules.
    We also compare the new scores with the previously recorded common results.
    These are internal consistency checks, not independent validation of labels.
    An approximate proximity diagnostic uses explicit nanosecond timestamps;
    it never removes nearby events or changes the model.
    ''', '''
    checks = {}
    years = {"train": {2016, 2017, 2018, 2019}, "validation": {2020}, "test": {2021}, "stress": {2022}}
    for split, frame in raw_clean.items():
        checks[f"{split}: Source2, years and unique events"] = bool(frame.Source.eq("Source2").all()
            and set(frame._t.dt.year.unique()) <= years[split] and frame._event.is_unique)
        checks[f"{split}: 34 inputs, target excluded"] = X[split].shape[1] == 34 and not set(X[split]) & FORBIDDEN
        for column in category_prep.vocab:
            assert X[split][column].cat.categories.equals(X["train"][column].cat.categories)
    for first, second in itertools.combinations(raw_clean, 2):
        checks[f"{first}/{second}: no temporal/event overlap"] = bool(
            raw_clean[first]._t.max() < raw_clean[second]._t.min()
            and not np.intersect1d(raw_clean[first]._event, raw_clean[second]._event).size)
    checks["Expected sample sizes"] = [len(y[s]) for s in y] == [500000, 50000, 99999, 50000]
    checks["Turning_Loop retained and constant"] = bool(X["train"].Turning_Loop.eq(0).all())
    checks["Complete original and derived dictionary"] = len(dictionary) == len(preview.columns) + 4
    frozen = json.loads((COMPARISONS / "frozen_selection.json").read_text())
    checks["Selection used validation only"] = frozen["test_used_for_selection"] is False
    references = {r["family"]: r for r in frozen["selected"]}
    for family, name in [("hgb", "Boosting"), ("rf", "Forest")]:
        actual = validation_scores[validation_scores.Model.eq(name)].iloc[0]
        checks[f"{name}: recorded validation reproduced"] = bool(
            np.isclose(actual.AP, references[family]["ap"], atol=1e-8, rtol=0)
            and np.isclose(thresholds[name], references[family]["threshold"], atol=1e-8, rtol=0))
    probe = raw_clean["train"].iloc[:50000].copy()
    ns_keys = nearby_keys(probe.assign(Start_Time=probe._t.astype("datetime64[ns]")))
    us_keys = nearby_keys(probe.assign(Start_Time=probe._t.astype("datetime64[us]")))
    checks["Proximity independent of timestamp unit"] = bool(np.array_equal(ns_keys, us_keys))
    check_table = pd.DataFrame({"Check": checks.keys(), "Passed": checks.values()})
    check_table.to_csv(REPORTS / "consistency_checks.csv", index=False)
    assert check_table.Passed.all(), check_table.loc[~check_table.Passed]
    display(Markdown("The period, preparation and reproduction checks passed. Full details are saved in the reports folder."))
    ''', '''
    The notebook uses `traffic_impact/` modules, the requirements files and the
    small common-comparison tables under `results/final_notebook_comparison/`.
    Keep that structure when reproducing it. Put the CSV in `data/` or set
    CSV_PATH, then run all cells in order. Caches, generated reports and figures
    are stored in `data/final_notebook/` and can be regenerated. Do not submit the
    raw CSV, caches or fitted models.

    Internal checks support reproducibility within this setup; they do not prove
    the absence of every possible bias. The complete comparison exports remain
    available for inspecting the wider search and its limits.
    ''')
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    for index, cell in enumerate(notebook.cells):
        cell.id = f"final-{index:03d}"
        assert chr(0x2014) not in cell.source
        if cell.cell_type == "code":
            compile(cell.source, f"final_cell_{index}", "exec")
    nbformat.validate(notebook)
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=TARGET)
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(f"Existing notebook preserved: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build(), output)
    print(f"Created {output.name}")
