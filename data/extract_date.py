import json

with open("/teamspace/studios/this_studio/TraderAgent/data_aggregated_v4/stock_values.json", "r") as f:
    data = json.load(f)

unique_dates = set()

for records in data.values():
    for record in records:
        unique_dates.add(record["Date"].split(" ")[0])

# Sort dates
unique_dates = sorted(unique_dates)

# Write to file
with open("dates.txt", "w") as f:
    for d in unique_dates:
        f.write(d + "\n")


import json

# Load master dates
with open("dates.txt", "r") as f:
    master_dates = {line.strip() for line in f if line.strip()}

# Load stock data
with open("/teamspace/studios/this_studio/TraderAgent/data_aggregated_v4/stock_values.json", "r") as f:
    data = json.load(f)

missing_report = {}

for ticker, records in data.items():
    ticker_dates = {r["Date"].split(" ")[0] for r in records}
    missing = sorted(master_dates - ticker_dates)

    if missing:
        missing_report[ticker] = missing

# Output results
if not missing_report:
    print("✅ All tickers have data for all dates in dates.txt")
else:
    print("❌ Missing dates found:\n")
    for ticker, dates in missing_report.items():
        print(f"{ticker} missing {len(dates)} dates:")
        for d in dates:
            print(f"  - {d}")
        print()
