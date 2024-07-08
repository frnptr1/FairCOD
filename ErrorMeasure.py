import numpy as np
import pandas as pd
from typing import Optional

# def evaluate_WeightedMeanAbsoluteError(df: pd.DataFrame, column_name_true: str, column_name_pred) -> tuple[float, float]:
def evaluate_WeightedMeanAbsoluteError(df: pd.DataFrame, column_name_true: str, column_name_pred:str, backlog:Optional[str], dpc:Optional[str]) -> None:

    classes_values = [1, 2, 3, 4,5,6,7,8,9]
    weights = [0, 1, 1, 2, 3, 5, 8, 19, 60]

    n = df.shape[0]
    total_error = 0
    max_error = sum(weights)  # This is 99 for your specific case

    print(f"{'i':^3} | {'LPM':^7} | {'Epic Name':^40} | {'True CoD':^10} | {'Pred CoD':^10} | {'Error':^7} | {'Total Error':^13}")

    for i, (index, row) in enumerate(df.iterrows()):

        true_idx = classes_values.index(row[column_name_true])
        pred_idx = classes_values.index(row[column_name_pred])

        if true_idx == pred_idx:
            continue

        start, end = min(true_idx, pred_idx), max(true_idx, pred_idx)
        error = sum(weights[start:end])
        total_error += error

        print(f"{i:^3d} | {str(row.name):^7} | {row['Epic_Name'][:40]:^40} | {int(row[column_name_true]):^10d} | {int(row[column_name_pred]):^10d} | {int(error):^7d} | {int(total_error):^13d}")

    print('-'*103)
    wmae = total_error / n
    normalized_wmae = wmae / max_error

    if backlog and dpc:

        print(f"{dpc:^7} | 'Backlog: '{backlog:^30} | 'WMAE: '{float(wmae):^8.2f} | 'N_WMAE: '{normalized_wmae:^8.2f} \n")

    else:
        print(f"'WMAE: '{float(wmae):^8.2f} | 'N_WMAE: '{normalized_wmae:^8.2f} \n")


    return None