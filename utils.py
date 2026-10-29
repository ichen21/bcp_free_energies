import numpy as np
import csv
import pandas as pd

def csv_to_array(data_file):
    # Read CSV robustly: try pandas first (fast C engine). If it raises a
    # parsing error because rows have uneven numbers of fields, fall back to
    # the Python csv reader which tolerates irregular rows. Missing or
    # non-numeric entries are coerced to NaN and then filled with -1.
    try:
        df = pd.read_csv(data_file, header=None)
    except Exception:
        # fallback: use Python csv.reader to preserve rows of varying length
        rows = []
        with open(data_file, 'r', newline='') as fh:
            reader = csv.reader(fh)
            for row in reader:
                rows.append(row)
        if len(rows) == 0:
            return np.empty((0, 0))
        max_cols = max(len(r) for r in rows)
        padded = [r + [''] * (max_cols - len(r)) for r in rows]
        df = pd.DataFrame(padded)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(-1)
    return df.to_numpy()

def compile_csv(dataset):
    # compiled_energies = np.array([])
    # for data_file in dataset:
    #     if compiled_energies.shape[0] == 0:
    #         compiled_energies = np.array([csv_to_array(data_file)])[0]
    #     else:
    #          compiled_energies = np.hstack((compiled_energies, np.array([csv_to_array(data_file)])[0]))

    return np.hstack([csv_to_array(data_file) for data_file in dataset]).T

