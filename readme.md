
# FinDoc Extractor 

**Lightweight ML-powered financial document extraction system** for annual reports, prospectuses, and invoices.

## What It Does

- **Extracts financial data** from PDFs: company name, revenue, profit, assets, debt, employees
- **Normalizes values**: handles multiple currencies (INR, USD, EUR, GBP) and units (crore, million, lakh)
- **Validates** against business rules (profit ≤ revenue, no negative assets)
- **Detects anomalies** using Isolation Forest (statistical outliers)
- **Predicts review needs** using Random Forest (which extractions to manually verify)
- **Evaluates accuracy** against ground truth CSV

## Architecture
