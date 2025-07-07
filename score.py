import pandas as pd
import sklearn.metrics
import pandas.api.types

class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, normalize: bool=True, weights_column_name: str=None) -> float:
    '''
    Kaggle-style accuracy score wrapper.

    Parameters:
    - solution: pd.DataFrame with ground truth labels and row ids.
    - submission: pd.DataFrame with predicted labels and row ids.
    - row_id_column_name: str, the name of the id column to align rows.
    - normalize: bool, if True return fraction correct, else count correct.
    - weights_column_name: optional str, sample weights column in solution.

    Returns:
    - float accuracy score.
    '''

    # Check required columns exist
    if row_id_column_name not in solution.columns or row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"'{row_id_column_name}' column must be present in both solution and submission.")

    # Align both dataframes by id to avoid mismatch
    solution = solution.sort_values(by=row_id_column_name).reset_index(drop=True)
    submission = submission.sort_values(by=row_id_column_name).reset_index(drop=True)

    if not solution[row_id_column_name].equals(submission[row_id_column_name]):
        raise ParticipantVisibleError("ID columns do not match between solution and submission after sorting.")

    # Drop the ID column for scoring
    solution_labels = solution.drop(columns=[row_id_column_name])
    submission_labels = submission.drop(columns=[row_id_column_name])

    # Handle sample weights if given
    sample_weight = None
    if weights_column_name:
        if weights_column_name not in solution_labels.columns:
            raise ParticipantVisibleError(f"Sample weights column '{weights_column_name}' not found in solution.")
        sample_weight = solution_labels.pop(weights_column_name).values
        if not pandas.api.types.is_numeric_dtype(sample_weight):
            raise ParticipantVisibleError("Sample weights are not numeric.")

    # Validate columns count
    if not (len(submission_labels.columns) == 1 or len(submission_labels.columns) == len(solution_labels.columns)):
        raise ParticipantVisibleError(f"Submission has {len(submission_labels.columns)} columns; expected 1 or {len(solution_labels.columns)}.")

    # Convert to numpy for sklearn
    y_true = solution_labels.values
    y_pred = submission_labels.values

    return float(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python score.py solution.csv submission.csv id_column_name [weights_column_name]")
        sys.exit(1)

    solution_file = sys.argv[1]
    submission_file = sys.argv[2]
    id_col = sys.argv[3]
    weights_col = sys.argv[4] if len(sys.argv) > 4 else None

    sol_df = pd.read_csv(solution_file)
    sub_df = pd.read_csv(submission_file)

    try:
        acc = score(sol_df, sub_df, row_id_column_name=id_col, weights_column_name=weights_col)
        print(f"Kaggle-style accuracy: {acc:.4f}")
    except ParticipantVisibleError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
