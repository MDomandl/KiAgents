import json
import pandas as pd

def lade_json(pfad):
    with open(pfad, 'r', encoding='utf-8') as f:
        return json.load(f)

def lade_excel_chat_infos(pfad):
    df = pd.read_excel(pfad)
    return {str(row.iloc[0]).strip().lower(): str(row.iloc[1]).strip().lower()
            for _, row in df.iterrows() if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1])}
