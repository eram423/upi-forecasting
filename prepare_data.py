#!/usr/bin/env python3
"""
Build data/upi_monthly_cleaned.csv from the raw NPCI workbooks in data/raw/.

    python prepare_data.py

The 11 yearly NPCI "UPI product statistics" workbooks (FY2016-17 ... FY2026-27) each have one sheet with the columns
Month | No. of Banks live on UPI | Volume (In Mn.) | Value (In Cr.).
Numbers are stored as TEXT with Indian digit grouping ("1,33,460.72") and sometimes a trailing tab, so they are
converted to floats. No value is changed otherwise. Rows are sorted by month and checked for duplicates and gaps.
"""
import re
from pathlib import Path

import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "data" / "raw", HERE / "data" / "upi_monthly_cleaned.csv"


def to_number(x):
    return float(re.sub(r"[,\s]", "", str(x)))   # removes commas, spaces, tabs


rows = []
for f in sorted(RAW.glob("*.xlsx")):
    ws = openpyxl.load_workbook(f, read_only=True, data_only=True).active
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i == 0 or r[0] is None:               # skip header / blank rows
            continue
        rows.append({"month_str": str(r[0]).strip(), "banks": int(to_number(r[1])),
                     "volume_mn": to_number(r[2]), "value_cr": to_number(r[3]), "source_file": f.name})

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["month_str"], format="%B-%Y")
df = df.sort_values("date").reset_index(drop=True)

assert not df["date"].duplicated().any(), "a month appears in more than one file"
assert df["date"].diff().dropna().dt.days.between(28, 31).all(), "gap in the monthly series"
assert (df[["banks", "volume_mn", "value_cr"]] >= 0).all().all()

df[["date", "month_str", "banks", "volume_mn", "value_cr", "source_file"]].assign(
    date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(OUT, index=False)
print(f"{len(df)} months, {df['date'].min():%b %Y} - {df['date'].max():%b %Y} -> {OUT}")
