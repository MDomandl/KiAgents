#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

def _exact_pairs(dec_dir: Path):
    bt = { _date_from_name(p): p for p in dec_dir.glob("BT_*.json") if _date_from_name(p) }
    rn = { _date_from_name(p): p for p in dec_dir.glob("RUN_*.json") if _date_from_name(p) }
    common = sorted(set(bt) & set(rn))
    return [ (bt[d], rn[d], datetime.fromisoformat(d), datetime.fromisoformat(d)) for d in common ]

def _clean_weights(d, min_bps=0.0):
    if not d: return {}
    thr = float(min_bps)/10000.0
    return {k: v for k, v in d.items() if abs(float(v)) >= thr}

def _date_from_name(p: Path):
    m = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
    return m.group(0) if m else None

def _load(p):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def _to_float(d):
    out={}
    for k,v in (d or {}).items():
        try: out[str(k)] = float(v)
        except: pass
    return out

def _bps(x): return 10000.0*float(x)

def _infer_to(old_w, new_w):
    ks=set(old_w)|set(new_w)
    return 0.5*sum(abs(new_w.get(k,0.0)-old_w.get(k,0.0)) for k in ks)

def _cmp(bt_p: Path, run_p: Path, tol_bps=5.0, ignore_cash=False, args=None):
    bt=_load(bt_p); rn=_load(run_p)
    d_bt = bt.get("as_of") or _date_from_name(bt_p)
    d_rn = rn.get("as_of") or _date_from_name(run_p)

    # --- CLI-Optionen bereitstellen ---
    min_bps = float(getattr(args, "min_bps", 0.0) or 0.0)
    limit   = int(getattr(args, "limit", 12) or 12)
    csv_out = getattr(args, "csv", None)

    bt_old=_to_float(bt.get("old_weights")); bt_new=_to_float(bt.get("new_weights"))
    rn_old=_to_float(rn.get("old_weights")); rn_new=_to_float(rn.get("new_weights"))

    # Cleaning immer anwenden (min_bps=0 → no-op)
    bt_old = _clean_weights(bt_old, min_bps)
    bt_new = _clean_weights(bt_new, min_bps)
    rn_old = _clean_weights(rn_old, min_bps)
    rn_new = _clean_weights(rn_new, min_bps)

    if ignore_cash:
        for d in (bt_old,bt_new,rn_old,rn_new): d.pop("CASH", None)

    names_bt={k for k,v in bt_new.items() if v>0 or k in bt_old}
    names_rn={k for k,v in rn_new.items() if v>0 or k in rn_old}
    only_bt=sorted(names_bt-names_rn); only_rn=sorted(names_rn-names_bt)

    keys=sorted(set(bt_new)|set(rn_new))
    diffs=[abs(_bps(bt_new.get(k,0.0)-rn_new.get(k,0.0))) for k in keys]
    offenders=[k for k in keys if abs(_bps(bt_new.get(k,0.0)-rn_new.get(k,0.0)))>tol_bps]
    max_abs=max(diffs) if diffs else 0.0
    mean_abs=(sum(diffs)/len(diffs)) if diffs else 0.0
    l1_bps=_bps(sum(abs(bt_new.get(k,0.0)-rn_new.get(k,0.0)) for k in keys))

    bt_to=bt.get("turnover") or bt.get("turnover_eff") or _infer_to(bt_old,bt_new)
    rn_to=rn.get("turnover") or rn.get("turnover_eff") or _infer_to(rn_old,rn_new)
    to_diff_bps=_bps(bt_to-rn_to)

    print(f"\n=== BT: {bt_p.name} ({d_bt})  vs  RUN: {run_p.name} ({d_rn}) ===")
    if only_bt or only_rn:
        print("Names differ:")
        if only_bt: print("  Only in BT :", ", ".join(only_bt))
        if only_rn: print("  Only in RUN:", ", ".join(only_rn))
    else:
        print("Names: OK")

    if offenders:
        print(f"Weights: {len(offenders)} tickers > {tol_bps:.1f} bps")
        for k in sorted(offenders, key=lambda k: -abs(_bps(bt_new.get(k, 0.0) - rn_new.get(k, 0.0))))[:limit]:
            d = _bps(bt_new.get(k, 0.0) - rn_new.get(k, 0.0))
            print(f"  {k:<8} Δ={d:+.1f} bps  (BT={bt_new.get(k, 0.0):.4f}  RUN={rn_new.get(k, 0.0):.4f})")
    else:
        print(f"Weights: OK (max {max_abs:.1f} bps, mean {mean_abs:.1f} bps, L1 {l1_bps:.1f} bps)")

    if csv_out:
        import csv
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "bt_weight", "run_weight", "delta_bps"])
            for k in sorted(keys):
                d = _bps(bt_new.get(k, 0.0) - rn_new.get(k, 0.0))
                w.writerow([k, f"{bt_new.get(k, 0.0):.6f}", f"{rn_new.get(k, 0.0):.6f}", f"{d:.1f}"])
        print(f"(CSV) geschrieben: {csv_out}")

    print(f"Turnover: BT={bt_to:.4f}  RUN={rn_to:.4f}  Δ={to_diff_bps:+.1f} bps")
    return not(only_bt or only_rn or offenders or abs(to_diff_bps)>tol_bps)

def _nearest_pairs(dec_dir: Path, within_days=7):
    bt, rn = [], []
    for p in dec_dir.glob("*.json"):
        d=_date_from_name(p);
        if not d: continue
        if p.name.startswith("BT_"):  bt.append((datetime.fromisoformat(d), p))
        if p.name.startswith("RUN_"): rn.append((datetime.fromisoformat(d), p))
    bt.sort(); rn.sort()
    pairs=[]
    for rd, rp in rn:
        # finde BT mit minimaler |Δ| Tage
        candidate=min(bt, key=lambda x: abs((x[0]-rd).days)) if bt else None
        if candidate:
            bd, bp = candidate
            if abs((bd-rd).days) <= within_days:
                pairs.append((bp, rp, bd, rd))
    return pairs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bt",  type=str)
    ap.add_argument("--run", type=str)
    ap.add_argument("--dir", type=str, help="Ordner mit BT_/RUN_ Bundles")
    ap.add_argument("--pair-nearest", action="store_true", help="BT/RUN per nächstem Datum paaren")
    ap.add_argument("--days", type=int, default=7, help="max. Tagesabstand für --pair-nearest")
    ap.add_argument("--bps", type=float, default=5.0)
    ap.add_argument("--ignore-cash", action="store_true")
    ap.add_argument("--pair-exact", action="store_true", help="BT/RUN mit identischem as_of paaren")
    ap.add_argument("--min-bps", type=float, default=0.0, help="Gewichte < min-bps werden ignoriert (vor Vergleich)")
    ap.add_argument("--limit", type=int, default=12, help="Max. Anzahl Offenders in der Ausgabe")
    ap.add_argument("--csv", type=str, help="Pfad für CSV-Diff (Ticker, BT, RUN, Δbps)")

    args=ap.parse_args()

    # main():
    if args.dir and args.pair_exact:
        pairs = _exact_pairs(Path(args.dir))
        if not pairs:
            print("Keine exakt passenden BT/RUN-Paare gefunden.");
            sys.exit(1)
        ok_all = True
        for bp, rp, bd, rd in pairs[:10]:
            print(f"\n(Pair) exact: {bd.date()}")
            ok = _cmp(bp, rp, tol_bps=args.bps, ignore_cash=args.ignore_cash, args=args)
            ok_all = ok_all and ok
        sys.exit(0 if ok_all else 2)

    if args.dir and args.pair_nearest:
        pairs=_nearest_pairs(Path(args.dir), within_days=args.days)
        if not pairs:
            print("Keine passenden BT/RUN-Paare gefunden."); sys.exit(1)
        ok_all=True
        for bp, rp, bd, rd in pairs[:12]:
            print(f"\n(Pair) nearest: BT={bd.date()}  RUN={rd.date()}  Δ={(rd-bd).days} Tage")
            ok = _cmp(bp, rp, tol_bps=args.bps, ignore_cash=args.ignore_cash, args=args)
            ok_all = ok_all and ok
        sys.exit(0 if ok_all else 2)

    if args.bt and args.run:
        ok=_cmp(Path(args.bt), Path(args.run), tol_bps=args.bps, ignore_cash=args.ignore_cash, args=args)
        sys.exit(0 if ok else 2)

    print("Entweder --bt & --run ODER --dir --pair-nearest angeben."); sys.exit(2)

if __name__=="__main__":
    main()
