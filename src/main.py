# src/main.py
import time
from pathlib import Path

from .loaders import load_cardio, load_parkinsons, load_genomics
from .run_tabular import run_all_tabular

def main():
    print("Starting project (AdaBoost removed).")

    datasets = []
    try:
        df = load_cardio()
        datasets.append(("cardiovascular", df))
    except Exception as e:
        print("Cardio load error:", e)

    try:
        df = load_parkinsons()
        datasets.append(("parkinsons", df))
    except Exception as e:
        print("Parkinsons load error:", e)

    try:
        df = load_genomics()
        datasets.append(("genomics", df))
    except Exception as e:
        print("Genomics load error:", e)

    if not datasets:
        print("No datasets loaded. Put CSVs under data/tabular and data/genomics and re-run.")
        return

    t0 = time.time()
    run_all_tabular(datasets)
    print("Elapsed: {:.1f}s".format(time.time() - t0))

if __name__ == "__main__":
    main()
