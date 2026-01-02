import pandas as pd

def as_series(x, name="Close") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            return x.iloc[:, 0]
    return pd.Series(x, name=name)

def regime_decision(cfg, data_client, as_of_str: str) -> str | None:
    if not getattr(cfg, "require_above_sma", False):
        return None
    if data_client.sp500_above_200dma(as_of_str):
        return None
    return "sp500_below_200dma"