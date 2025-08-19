import pandas as pd

def as_series(x, name="Close") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            return x.iloc[:, 0]
    return pd.Series(x, name=name)
