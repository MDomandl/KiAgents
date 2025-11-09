from pathlib import Path
import json, sys, csv

def main(decision_json_path: str, out_csv_path: str, topk: int = 12):
    p = Path(decision_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    weights = data.get("weights") or data.get("positions") or {}
    # Fallback: manche Bundles haben 'holdings' als Liste
    if not weights and "holdings" in data:
        weights = {h.get("ticker") or h.get("symbol"): h.get("weight", 0.0)
                   for h in data["holdings"]}

    # rudimentäre Meta-Felder, wenn vorhanden
    ranks  = {k: v for k, v in data.get("ranks", {}).items()} if "ranks" in data else {}
    scores = {k: v for k, v in data.get("scores", {}).items()} if "scores" in data else {}
    vols   = {k: v for k, v in data.get("vol", {}).items()}    if "vol" in data else {}

    # TopK nach Absolutgewicht
    items = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:topk]

    out = Path(out_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Ticker","Weight","Rank","Score","Vol"])
        for t, wgt in items:
            w.writerow([t, f"{wgt:.6f}", ranks.get(t, ""), scores.get(t, ""), vols.get(t, "")])

    print(f"[OK] Wrote TopK CSV → {out}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m aktien_oop.tools.extract_topk_from_decision <decision.json> <out.csv> [topk]")
        sys.exit(1)
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    main(sys.argv[1], sys.argv[2], topk)
