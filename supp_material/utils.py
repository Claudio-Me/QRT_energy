import pandas as pd
import os
import numpy as np
from datetime import datetime

def save_results_to_csv(dataset_wit_IDs, predictions, file_name: str=None):
    # assume X_test (pd.DataFrame) and preds (array-like) are defined
    # choose ID column if present, otherwise use index
    id_col = 'ID' if 'ID' in dataset_wit_IDs.columns else None
    ids = dataset_wit_IDs[id_col].values if id_col else dataset_wit_IDs.index.values
    # ensure lengths match
    print(f"IDs length: {len(ids)}, Predictions length: {len(predictions)}")
    print(len(ids) == len(predictions))
    assert len(ids) == len(predictions), "Length mismatch between IDs and predictions"

    # Ensure 1D
    preds = np.asarray(predictions)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.ravel()

    out_df = pd.DataFrame({'ID': ids, 'TARGET': preds})

    # ensure output dir exists (current dir used here)
    out_dir = '.'
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{file_name}_{ts}.csv"
    out_path = os.path.join(out_dir, fname)

    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to `{out_path}`")