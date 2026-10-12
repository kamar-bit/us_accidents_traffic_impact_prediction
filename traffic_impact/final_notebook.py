"""Data access and column definitions for the final teaching notebook.

Model preparation and training remain visible in the notebook. This module
handles file integrity, reusable samples, and the complete data dictionary.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import prepare
from .experiment import POI, WNUM, WCAT, LIGHT
from .paths import ROOT

LIMITS = {"train": 500010, "validation": 50000, "test": 100000, "stress": 50000}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_csv():
    candidates = [ROOT / "data/US_Accidents_March23.csv",
                  ROOT.parent.parent / "US_Accidents_March23.csv"]
    metadata = ROOT / "results/audit/audit.json"
    if metadata.exists():
        candidates.append(Path(json.loads(metadata.read_text(encoding="utf-8"))["file"]))
    return next((p for p in candidates if p.is_file()), candidates[0])


def load_samples(csv_path, work):
    """Reuse verified local samples, or reconstruct them from the original CSV.

    The separate experiment cache can be read without being modified. A new
    cache is generated under work when the file, code, environment or hashes
    differ. The return value includes the audit folder used for the sample.
    """
    csv_path, work = Path(csv_path), Path(work)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Download US_Accidents_March23.csv or set CSV_PATH: {csv_path}")
    signature = {"csv": str(csv_path.resolve()), "bytes": csv_path.stat().st_size,
                 "mtime_ns": csv_path.stat().st_mtime_ns,
                 "prepare_sha256": sha256(prepare.__file__), "pandas": pd.__version__,
                 "numpy": np.__version__, "limits": LIMITS}
    python_version = list(sys.version_info[:2])
    for candidate in [work, ROOT / "data/final_notebook_experiments"]:
        stamp = candidate / "cache_manifest.json"
        audit_folder = candidate / "results/audit"
        if not stamp.exists() or not (audit_folder / "audit.json").exists():
            continue
        manifest = json.loads(stamp.read_text(encoding="utf-8"))
        same_python = manifest.get("python") == python_version
        if candidate != work:
            protocol_file = ROOT / "results/final_notebook_comparison/protocol.json"
            if protocol_file.exists():
                protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
                version = protocol.get("versions", {}).get("python", "")
                same_python = version.startswith(".".join(map(str, python_version)) + ".")
        paths = {split: candidate / "data/cache" / f"{split}.pkl" for split in LIMITS}
        valid = manifest.get("signature") == signature and same_python
        valid = valid and all(p.is_file() and sha256(p) == manifest.get("sha", {}).get(s)
                              for s, p in paths.items())
        if valid:
            try:
                frames = {s: pd.read_pickle(p) for s, p in paths.items()}
            except (NotImplementedError, ValueError, TypeError, ModuleNotFoundError):
                continue
            return frames, json.loads((audit_folder / "audit.json").read_text()), audit_folder
    work.mkdir(parents=True, exist_ok=True)
    messages = io.StringIO()
    with redirect_stdout(messages):
        audit = prepare.run(csv_path, out=work, limits=LIMITS)
    (work / "preparation.log").write_text(messages.getvalue(), encoding="utf-8")
    paths = {s: work / "data/cache" / f"{s}.pkl" for s in LIMITS}
    (work / "cache_manifest.json").write_text(json.dumps({"signature": signature,
        "python": python_version, "sha": {s: sha256(p) for s, p in paths.items()}}, indent=2),
        encoding="utf-8")
    return {s: pd.read_pickle(p) for s, p in paths.items()}, audit, work / "results/audit"


def nearby_keys(frame):
    """Approximate proximity, not event identity, with a fixed timestamp unit."""
    time = pd.to_datetime(frame.Start_Time, format="mixed", errors="coerce")
    if time.isna().any():
        raise ValueError("Proximity diagnostic requires valid start timestamps")
    bins = pd.DataFrame({"time": time.astype("datetime64[ns]").astype("int64") // (15 * 60 * 10**9),
        "latitude": np.floor(frame.Start_Lat * 1000),
        "longitude": np.floor(frame.Start_Lng * 1000)}, index=frame.index)
    return pd.util.hash_pandas_object(bins, index=False).to_numpy()


def data_dictionary(columns):
    """Define every original field, including exclusions, then derived inputs."""
    rows = {}

    def add(name, meaning, kind, unit, role, availability, treatment):
        rows[name] = {"Column": name, "Meaning": meaning, "Type": kind,
                      "Unit / values": unit, "Role": role,
                      "Time reference / availability": availability, "Treatment": treatment}

    add("ID", "Record identifier", "text", "identifier", "audit", "record metadata",
        "Not a predictor; an event may have several record identifiers")
    add("Source", "Source identifier for the record", "category", "Source1/2/3", "filter",
        "record metadata", "Keep Source2, then exclude from predictors")
    add("Severity", "Recorded impact on traffic", "ordinal", "1, 2, 3, 4", "target",
        "assignment time unknown", "Positive if 3 or 4; never an input")
    add("Start_Time", "Recorded accident start", "datetime", "local date/time", "derive/audit",
        "start proxy", "Parse date; derive hour, weekday, month and weekend")
    add("End_Time", "Recorded event end", "datetime", "local date/time", "audit only",
        "end; initial availability not established", "Event identity only, not a predictor")
    for name, meaning, unit in [("Start_Lat", "Start latitude", "degrees, -90 to 90"),
                                ("Start_Lng", "Start longitude", "degrees, -180 to 180")]:
        add(name, meaning, "numeric", unit, "input", "start context, publication unknown",
            "Keep; invalid coordinates become missing")
    for name in ["End_Lat", "End_Lng"]:
        add(name, "End coordinate of the recorded affected segment", "numeric", "degrees", "excluded",
            "initial availability not established", "Not used for prediction")
    add("Distance(mi)", "Recorded length of the affected road segment", "numeric", "miles", "audit only",
        "initial availability not established", "Event identity only, not a predictor")
    add("Description", "Text description of the accident", "text", "free text", "audit only",
        "version/publication time unknown", "Event identity only; no text modelling")
    for name, meaning in [("Street", "Street name"), ("City", "City"), ("County", "County"),
                          ("Zipcode", "Postal code"), ("Airport_Code", "Associated weather station code")]:
        add(name, meaning, "category", "text code/name", "excluded", "associated record context",
            "Omitted to keep the location representation manageable")
    add("State", "US state", "category", "state abbreviation", "input", "start location",
        "Training vocabulary; native category or one-hot")
    add("Country", "Country", "category", "US in this file", "excluded", "start location",
        "Constant in the audited file")
    add("Timezone", "Time zone at the accident location", "category", "zone label", "input",
        "start location", "Training vocabulary; Start_Time is already local")
    add("Weather_Timestamp", "Time of the associated weather observation", "datetime", "date/time",
        "availability check", "observation time; publication time unknown",
        "Accept from -120 to +60 minutes, or -120 to 0 in strict scenario; not an input")
    weather = {"Temperature(F)": ("Air temperature", "degrees Fahrenheit"),
        "Wind_Chill(F)": ("Wind chill temperature", "degrees Fahrenheit"),
        "Humidity(%)": ("Relative humidity", "percent, 0 to 100"),
        "Pressure(in)": ("Air pressure", "inches of mercury"),
        "Visibility(mi)": ("Visibility distance", "miles"),
        "Wind_Speed(mph)": ("Wind speed", "miles per hour"),
        "Precipitation(in)": ("Recorded precipitation amount", "inches"),
        "Wind_Direction": ("Reported wind direction", "compass labels, Calm, Variable"),
        "Weather_Condition": ("Reported weather condition", "text categories")}
    for name, (meaning, unit) in weather.items():
        treatment = ("Harmonise spelling; retain distinct compass directions" if name == "Wind_Direction"
                     else "Retain up to 25 frequent training labels; rare/unseen -> Other"
                     if name == "Weather_Condition" else
                     "Numeric; mask invalid bounds where defined; native missingness or train median")
        add(name, meaning, "category" if name in WCAT else "numeric", unit, "input",
            "associated weather timestamp; masked outside scenario", treatment)
    road = {"Amenity": "Nearby amenity", "Bump": "Nearby road bump", "Crossing": "Nearby crossing",
        "Give_Way": "Nearby give-way sign", "Junction": "Nearby road junction",
        "No_Exit": "Nearby no-exit road", "Railway": "Nearby railway",
        "Roundabout": "Nearby roundabout", "Station": "Nearby station",
        "Stop": "Nearby stop sign", "Traffic_Calming": "Nearby traffic calming",
        "Traffic_Signal": "Nearby traffic signal", "Turning_Loop": "Nearby turning loop"}
    for name, meaning in road.items():
        constant = name == "Turning_Loop"
        add(name, meaning + " indicator (not the accident's cause)", "boolean", "False/True",
            "excluded" if constant else "input", "location context; update time unknown",
            "Excluded: always False in this file, no learnable signal" if constant else "0/1; keep")
    for name, meaning in [("Sunrise_Sunset", "Day/night relative to sunrise and sunset"),
                          ("Civil_Twilight", "Day/night relative to civil twilight (Sun at -6 degrees)"),
                          ("Nautical_Twilight", "Day/night relative to nautical twilight (-12 degrees)"),
                          ("Astronomical_Twilight", "Day/night relative to astronomical twilight (-18 degrees)")]:
        add(name, meaning, "category", "Day/Night", "input", "accident start",
            "Keep separate; missing -> Missing; not artificial road lighting")
    missing = set(columns) - set(rows)
    if missing:
        raise ValueError(f"Undefined original columns: {sorted(missing)}")
    derived = []
    for name, meaning, unit in [("hour", "Local start hour", "0 to 23"),
                                ("day_of_week", "Local start weekday", "0=Monday to 6=Sunday"),
                                ("month", "Start month", "1 to 12"),
                                ("weekend", "Saturday or Sunday indicator", "0/1")]:
        add(name, meaning, "integer", unit, "derived input", "Start_Time", "Derived without target")
        derived.append(name)
    return pd.DataFrame([rows[c] for c in list(columns) + derived])
