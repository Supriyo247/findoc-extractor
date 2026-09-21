
"""FinDoc Extractor - Minimal LLM-free version with ML quality control."""
import re, json, csv, hashlib, numpy as np
from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import pdfplumber

# =============== SCHEMAS & UTILS ===============

UNIT_MAP = {
    "units": 1.0, "thousand": 1e3, "lakh": 1e5,
    "million": 1e6, "crore": 1e7, "billion": 1e9,
    "crores": "crore", "cr": "crore", "lakhs": "lakh", "lacs": "lakh",
    "mn": "million", "mm": "million", "bn": "billion", "thousands": "thousand",
}

CURRENCIES = {"INR", "USD", "EUR", "GBP"}

def to_number(val):
    if val is None or isinstance(val, bool):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"null", "none", "n/a", "-", "nil"}:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d.\-]", "", s)
    if s in {"", "-", "."}:
        return None
    try:
        n = float(s)
        return -abs(n) if negative else n
    except:
        return None

def canonical_unit(u):
    if u is None:
        return None
    s = str(u).strip().lower()
    s = re.sub(r"\b(in|of|rs|inr|usd|eur|gbp)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s).strip()
    if s in UNIT_MAP and isinstance(UNIT_MAP[s], (int, float)):
        return s
    for token in s.split():
        if token in UNIT_MAP:
            val = UNIT_MAP[token]
            return val if isinstance(val, (int, float)) else token
    return None

def normalize(rec):
    """Clean record + calculate absolute values."""
    out = dict(rec)
    unit = canonical_unit(rec.get("unit"))
    out["unit"] = unit
    mult = UNIT_MAP.get(unit) if isinstance(UNIT_MAP.get(unit), (int, float)) else None
    
    for f in ["revenue", "net_profit", "total_assets", "total_debt"]:
        n = to_number(rec.get(f))
        out[f] = n
        out[f"{f}_abs"] = n * mult if n and mult else None
    
    fy = to_number(rec.get("fiscal_year"))
    out["fiscal_year"] = int(fy) if fy else None
    
    emp = to_number(rec.get("employees"))
    out["employees"] = int(emp) if emp else None
    
    cur = rec.get("currency")
    out["currency"] = str(cur).upper() if cur else None
    
    rev, prof = out.get("revenue_abs"), out.get("net_profit_abs")
    out["net_margin"] = (prof / rev) if rev and prof is not None else None
    return out

# =============== EXTRACTION ===============

_NUM = r"(-?\(?[\d,]+(?:\.\d+)?\)?)"
_LABELS = {
    "revenue": ["revenue from operations", "total revenue", "net sales", "revenue"],
    "net_profit": ["net profit", "profit after tax", "net income", "profit for the year", "profit:"],
    "total_assets": ["total assets", "assets:"],
    "total_debt": ["total debt", "total borrowings", "borrowings", "debt:"],
}
_COMPANY = re.compile(r"^[A-Z][\w&.,'\- ]*?\b(?:Limited|Ltd|Inc|Corporation|Corp|LLC|PLC)(?=\W|$)", re.M)

def extract_rules(text):
    """Regex-based extraction (no API needed)."""
    rec = {
        "company_name": None, "document_type": None, "fiscal_year": None,
        "currency": None, "unit": None, "revenue": None, "net_profit": None,
        "total_assets": None, "total_debt": None, "employees": None,
        "top_risks": [], "confidence_scores": {}
    }
    
    m = _COMPANY.search(text)
    rec["company_name"] = m.group(0).strip() if m else None
    rec["confidence_scores"]["company_name"] = 1.0 if m else 0.0
    
    low = text.lower()
    if "prospectus" in low:
        rec["document_type"] = "ipo prospectus"
    elif "annual report" in low:
        rec["document_type"] = "annual report"
    elif "quarter" in low:
        rec["document_type"] = "quarterly report"
    else:
        rec["document_type"] = "other"
    
    m = re.search(r"(?:year ended|annual report|report\s+).*?(20\d{2})", text, re.I)
    if m:
        rec["fiscal_year"] = int(m.group(1))
    
    m = re.search(r"\b(crore|lakh|million|billion|thousand)s?\b", low)
    rec["unit"] = m.group(1) if m else None
    
    if re.search(r"\bINR\b|₹", text):
        rec["currency"] = "INR"
    elif re.search(r"\bUSD\b|\$", text):
        rec["currency"] = "USD"
    elif re.search(r"\bEUR\b|€", text):
        rec["currency"] = "EUR"
    
    for field, labels in _LABELS.items():
        for label in labels:
            m = re.search(rf"{re.escape(label)}[^\n\d(]{{0,40}}?{_NUM}", text, re.I)
            if m:
                rec[field] = m.group(1)
                rec["confidence_scores"][field] = 0.85
                break
    
    m = re.search(r"employees?:?\s*([\d,]+)", text, re.I)
    if m:
        rec["employees"] = m.group(1)
        rec["confidence_scores"]["employees"] = 0.8
    
    return rec

# =============== VALIDATION & ML ===============

def validate(rec):
    """Business rule validation."""
    issues = []
    
    for f in ["company_name", "fiscal_year", "revenue", "net_profit"]:
        if rec.get(f) in (None, ""):
            issues.append({"field": f, "severity": "error", "message": f"{f} missing"})
    
    has_money = any(rec.get(f) is not None for f in ["revenue", "net_profit", "total_assets", "total_debt"])
    if has_money and rec.get("unit") not in UNIT_MAP:
        issues.append({"field": "unit", "severity": "error", "message": "unit missing/invalid"})
    
    if not rec.get("currency") and has_money:
        issues.append({"field": "currency", "severity": "warning", "message": "currency missing"})
    
    rev, prof = rec.get("revenue_abs"), rec.get("net_profit_abs")
    if rev and prof and prof > rev:
        issues.append({"field": "net_profit", "severity": "error", "message": "profit > revenue"})
    
    return issues

class AnomalyDetector:
    def __init__(self):
        self.model = None
    
    def fit(self, records):
        X = []
        for r in records:
            rev, assets, debt = r.get("revenue_abs"), r.get("total_assets_abs"), r.get("total_debt_abs")
            prof = r.get("net_profit_abs")
            emp = r.get("employees", 1)
            
            X.append([
                (debt / assets) if assets else 0.5,
                (prof / rev) if rev and prof is not None else 0.1,
                (rev / emp) if rev and emp else 1e6,
            ])
        
        if len(X) >= 5:
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.model.fit(X)
    
    def predict(self, record):
        if not self.model:
            return False, 0.0
        rev, assets, debt = record.get("revenue_abs"), record.get("total_assets_abs"), record.get("total_debt_abs")
        prof = record.get("net_profit_abs")
        emp = record.get("employees", 1)
        
        X = [[
            (debt / assets) if assets else 0.5,
            (prof / rev) if rev and prof is not None else 0.1,
            (rev / emp) if rev and emp else 1e6,
        ]]
        pred = self.model.predict(X)[0]
        score = self.model.score_samples(X)[0]
        return pred == -1, float(score)

class ReviewPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def fit(self, df):
        X, y = [], []
        for _, r in df.iterrows():
            X.append([
                r.get("avg_confidence", 0.7),
                1 if pd.isna(r.get("revenue_abs")) else 0,
                1 if pd.isna(r.get("net_profit_abs")) else 0,
                1 if r.get("fiscal_year") is None else 0,
            ])
            y.append(1 if r.get("status") == "review" else 0)
        
        if len(set(y)) >= 2 and len(X) >= 5:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight="balanced")
            self.model.fit(X, y)
    
    def predict(self, record, avg_conf):
        if not self.model:
            return 0.0
        X = [[
            avg_conf,
            1 if record.get("revenue") is None else 0,
            1 if record.get("net_profit") is None else 0,
            1 if record.get("fiscal_year") is None else 0,
        ]]
        X = self.scaler.transform(X)
        return float(self.model.predict_proba(X)[0][1])

# =============== PDF HANDLING ===============

def read_pdf(path):
    """Extract text from PDF."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages:
            pages.append(pg.extract_text() or "")
    return pages

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

# =============== DATABASE ===============

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extractions (
            id INTEGER PRIMARY KEY,
            filename TEXT, sha256 TEXT, engine TEXT,
            company_name TEXT, fiscal_year INTEGER, currency TEXT, unit TEXT,
            revenue REAL, net_profit REAL, total_assets REAL, total_debt REAL,
            revenue_abs REAL, net_profit_abs REAL, total_assets_abs REAL, total_debt_abs REAL,
            net_margin REAL, employees INTEGER,
            avg_confidence REAL, anomaly_score REAL, review_prob REAL,
            status TEXT, extracted_at TEXT, UNIQUE(sha256, engine)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS validation_issues (
            id INTEGER PRIMARY KEY,
            extraction_id INTEGER,
            field TEXT, severity TEXT, message TEXT,
            FOREIGN KEY(extraction_id) REFERENCES extractions(id)
        )
    """)
    conn.commit()
    return conn

def save_extraction(conn, filename, sha, engine, record, issues, status, avg_conf, anom_score, review_prob):
    conn.execute("DELETE FROM extractions WHERE sha256=? AND engine=?", (sha, engine))
    cur = conn.execute(f"""
        INSERT INTO extractions (
            filename, sha256, engine, company_name, fiscal_year, currency, unit,
            revenue, net_profit, total_assets, total_debt,
            revenue_abs, net_profit_abs, total_assets_abs, total_debt_abs,
            net_margin, employees, avg_confidence, anomaly_score, review_prob,
            status, extracted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, sha, engine, record.get("company_name"), record.get("fiscal_year"),
        record.get("currency"), record.get("unit"),
        record.get("revenue"), record.get("net_profit"), record.get("total_assets"), record.get("total_debt"),
        record.get("revenue_abs"), record.get("net_profit_abs"), record.get("total_assets_abs"), record.get("total_debt_abs"),
        record.get("net_margin"), record.get("employees"), avg_conf, anom_score, review_prob,
        status, datetime.now().isoformat()
    ))
    ext_id = cur.lastrowid
    for issue in issues:
        conn.execute(
            "INSERT INTO validation_issues (extraction_id, field, severity, message) VALUES (?, ?, ?, ?)",
            (ext_id, issue["field"], issue["severity"], issue["message"])
        )
    conn.commit()

# =============== MAIN PIPELINE ===============

def main():
    Path("data").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Generate sample PDFs
    generate_samples()
    
    conn = init_db("data/findoc.db")
    anomaly = AnomalyDetector()
    reviewer = ReviewPredictor()
    
    # Process PDFs
    counts = {"ok": 0, "review": 0}
    for pdf_path in sorted(Path("data/pdfs").glob("*.pdf")):
        text = "\n".join(read_pdf(pdf_path))
        raw = extract_rules(text)
        record = normalize(raw)
        issues = validate(record)
        
        confidences = list(raw.get("confidence_scores", {}).values())
        avg_conf = np.mean(confidences) if confidences else 0.7
        
        status = "review" if issues else "ok"
        is_anom, anom_score = anomaly.predict(record)
        if is_anom:
            status = "review"
            issues.append({"field": "overall", "severity": "warning", "message": "Anomaly detected"})
        
        review_prob = reviewer.predict(record, avg_conf)
        
        save_extraction(conn, pdf_path.name, sha256(pdf_path), "rules", record, issues, status, avg_conf, anom_score, review_prob)
        counts[status] += 1
        print(f"✓ {pdf_path.name}: {status}")
    
    # Export results
    df = pd.read_sql("SELECT * FROM extractions", conn)
    if len(df) >= 5:
        anomaly.fit(df.to_dict("records"))
        reviewer.fit(df)
    
    df.to_csv("reports/extractions.csv", index=False)
    print(f"\n✓ Processed {len(df)} PDFs: {counts['ok']} ok, {counts['review']} review")
    print(f"  Results: reports/extractions.csv")

def generate_samples():
    """Create 6 synthetic financial PDFs."""
    Path("data/pdfs").mkdir(exist_ok=True)
    
    companies = [
        ("zenith_2024.pdf", "Zenith Textiles Ltd", 2024, "INR", "crore", "1234.50", "98.20", "2050.00", "410.75", "4520"),
        ("brightpath_2023.pdf", "BrightPath Software Inc", 2023, "USD", "million", "845.3", "121.9", "1560.2", "230.0", "3100"),
        ("kaveri_2024.pdf", "Kaveri Foods Ltd", 2024, "INR", "crore", "412.80", "27.45", "530.10", "96.40", "1180"),
        ("northwind_2022.pdf", "Northwind Logistics", 2022, "EUR", "million", "2310.0", "-45.6", "3800.5", "1205.0", "12400"),
        ("solaris_2025.pdf", "Solaris Energy Ltd", 2025, "INR", "lakh", "88450", "6120", "120300", "35200", "640"),
        ("harbor_2024.pdf", "Harbor Retail PLC", 2024, "GBP", "thousand", "567890", "23410", "410220", "88900", "7250"),
    ]
    
    for fname, name, year, cur, unit, rev, prof, assets, debt, emp in companies:
        c = canvas.Canvas(f"data/pdfs/{fname}", pagesize=A4)
        c.drawString(50, 750, name)
        c.drawString(50, 700, f"Annual Report {year} ({cur} {unit})")
        c.drawString(50, 650, f"Revenue: {rev} | Profit: {prof} | Assets: {assets} | Debt: {debt} | Employees: {emp}")
        c.showPage()
        c.save()
    
    # Ground truth CSV
    with open("data/ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "company_name", "fiscal_year", "currency", "unit", "revenue", "net_profit", "total_assets", "total_debt", "employees"])
        w.writeheader()
        for fname, name, year, cur, unit, rev, prof, assets, debt, emp in companies:
            w.writerow({
                "filename": fname, "company_name": name, "fiscal_year": year,
                "currency": cur, "unit": unit, "revenue": rev, "net_profit": prof,
                "total_assets": assets, "total_debt": debt, "employees": emp
            })

if __name__ == "__main__":
    main()