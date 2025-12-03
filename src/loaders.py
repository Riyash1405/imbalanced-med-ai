# src/loaders.py
from pathlib import Path
import pandas as pd
import csv

DATA_ROOT = Path.cwd() / "data"

def _detect_sep(path: Path, n_lines: int = 5):
    """Return separator by checking first lines for commas vs semicolons."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    # look at first chunk
    first = "\n".join(text.splitlines()[:n_lines])
    # count occurrences
    commas = first.count(",")
    semis = first.count(";")
    # prefer semicolon if more semicolons than commas in sample
    if semis > commas:
        return ";"
    if commas > semis:
        return ","
    # fallback to csv.Sniffer
    try:
        dialect = csv.Sniffer().sniff(first)
        return dialect.delimiter
    except Exception:
        return ","  # default

def _read_csv_try(path: Path):
    if not path.exists():
        return None
    sep = _detect_sep(path)
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        # try flexible engine
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            # as final attempt try no sep (single column) then split
            df = pd.read_csv(path, header=None, dtype=str)
            # if single column contains semicolons or commas, try splitting
            if df.shape[1] == 1:
                s = df.iloc[:, 0].astype(str)
                if s.str.contains(";").any():
                    return pd.read_csv(path, sep=";", engine="python")
                if s.str.contains(",").any():
                    return pd.read_csv(path, sep=",", engine="python")
            return None

def load_cardio():
    candidates = [
        DATA_ROOT / "tabular" / "cardio_train.csv",
        DATA_ROOT / "tabular" / "cardio.csv",
        DATA_ROOT / "cardio_train.csv",
        DATA_ROOT / "cardio.csv"
    ]
    for p in candidates:
        df = _read_csv_try(p)
        if df is None:
            continue
        # normalize names
        df.columns = [c.strip() for c in df.columns]
        low = [c.lower().strip() for c in df.columns]
        if "cardio" in low:
            orig = df.columns[low.index("cardio")]
            df = df.rename(columns={orig: "target"})
        elif "target" in low:
            orig = df.columns[low.index("target")]
            df = df.rename(columns={orig: "target"})
        else:
            # fallback assume last col is target
            df = df.rename(columns={df.columns[-1]: "target"})
        return df
    raise FileNotFoundError("Cardio CSV not found. Place cardio_train.csv in data/tabular or data/")

def load_parkinsons():
    p = DATA_ROOT / "tabular" / "parkinsons.csv"
    df = _read_csv_try(p)
    if df is None:
        raise FileNotFoundError(p)
    low = [c.lower().strip() for c in df.columns]
    if "status" in low:
        df = df.rename(columns={df.columns[low.index("status")]: "target"})
    elif "target" not in low:
        df = df.rename(columns={df.columns[-1]: "target"})
    return df

def load_genomics():
    p = DATA_ROOT / "genomics" / "genomics_matrix.csv"
    df = _read_csv_try(p)
    if df is None:
        raise FileNotFoundError(p)
    low = [c.lower().strip() for c in df.columns]
    if "label" in low and "target" not in low:
        df = df.rename(columns={df.columns[low.index("label")]: "target"})
    elif "target" not in low:
        df = df.rename(columns={df.columns[-1]: "target"})
    return df
