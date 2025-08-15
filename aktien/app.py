import streamlit as st
import pandas as pd
from helper import main as calculate_results

st.set_page_config(page_title="Clenow Momentum", layout="wide", initial_sidebar_state="collapsed")
st.title("Clenow Momentum Strategie")

def render_centered_table(df):
    # Platz hinzufügen
    df.insert(0, "Platz", range(1, len(df) + 1))

    # Spalten umbenennen (Deutsch)
    df = df.rename(columns={
        "ticker": "Ticker",
        "slope": "Steigung",
        "r2": "Bestimmtheitsmaß",
        "stop_loss_pct": "Stop-Loss (%)",
        "volatility": "Volatilität",
        "allocation_pct": "Kapitalgewichtung (%)"
    })

    # Werte runden
    df["Steigung"] = df["Steigung"].round(4)
    df["Bestimmtheitsmaß"] = df["Bestimmtheitsmaß"].round(4)
    df["Stop-Loss (%)"] = df["Stop-Loss (%)"].round(4)
    df["Volatilität"] = df["Volatilität"].round(2)
    df["Kapitalgewichtung (%)"] = df["Kapitalgewichtung (%)"].round(2)

    # Styling zentriert
    styled_df = df.style.set_table_styles(
        [{
            'selector': 'th, td',
            'props': [('text-align', 'center')]
        }]
    ).set_properties(**{'text-align': 'center'})

    st.dataframe(styled_df, use_container_width=True)

# Analyse starten
if st.button("Analyse starten"):
    with st.spinner("Daten werden geladen..."):
        df = calculate_results()
        if df is not None and not df.empty:
            st.success("Analyse abgeschlossen.")
            st.subheader("Top 8 Momentum-Aktien nach Clenow + Filter + Risiko-Allokation:")
            render_centered_table(df)

            st.markdown("""
            ---
            **Strategie-Regeln:**

            - **Neubewertung:** Jeden Mittwoch. Ist eine Aktie nicht mehr im Score, wird sie vollständig verkauft.
            - **Neuplatzierung Stop-Loss:** Jeden Mittwoch basierend auf dem aktuellen ATR.
            - **Rebalancing:** Alle 6 Monate.
            """)
        else:
            st.warning("Kein Ergebnis. Prüfe Filter oder Marktlage.")
