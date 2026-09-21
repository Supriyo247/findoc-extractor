"""Simple evaluation against ground truth."""
import re, pandas as pd
from pathlib import Path

def norm_text(s):
    s = re.sub(r"[^a-z0-9 ]", "", str(s).lower())
    s = re.sub(r"\b(limited|ltd|inc|corp|corporation|llc|plc)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

def values_match(field, truth, pred):
    if pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return False
    if field == "company_name":
        return norm_text(truth) == norm_text(pred)
    if field == "currency":
        return str(truth).upper() == str(pred).upper()
    try:
        t, p = float(truth), float(pred)
        if t == 0:
            return p == 0
        return abs(p - t) / abs(t) <= 0.01
    except:
        return False

extracts = pd.read_csv("reports/extractions.csv", dtype=str)
truth = pd.read_csv("data/ground_truth.csv", dtype=str)

fields = ["company_name", "fiscal_year", "currency", "revenue_abs", "net_profit_abs", "total_assets_abs", "total_debt_abs", "employees"]
pred_by_file = {r["filename"]: r for r in extracts.to_dict("records")}

stats = {f: [0, 0] for f in fields}
for t in truth.to_dict("records"):
    pred = pred_by_file.get(t["filename"], {})
    for f in fields:
        tv = t.get(f)
        if tv and tv != "":
            stats[f][0] += 1
            if values_match(f, tv, pred.get(f)):
                stats[f][1] += 1

print("\n📊 ACCURACY vs Ground Truth:")
print(f"{'Field':<20} {'Tested':<8} {'Correct':<8} {'Accuracy':<10}")
print("-" * 50)
total_n, total_c = 0, 0
for f in fields:
    n, c = stats[f]
    if n:
        acc = f"{c/n:.0%}"
        print(f"{f:<20} {n:<8} {c:<8} {acc:<10}")
        total_n += n
        total_c += c

if total_n:
    print("-" * 50)
    print(f"{'OVERALL':<20} {total_n:<8} {total_c:<8} {total_c/total_n:.0%}")
