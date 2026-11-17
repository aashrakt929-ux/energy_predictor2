"""يولّد sample_data.csv: 36 شهرًا استهلاك كهرباء واقعي (موسمية صيفية + نمو بسيط + ارتفاعين غير معتادين)."""
import csv
import math
import random

random.seed(7)
rows = []
year, month = 2023, 11
for i in range(36):
    base = 320 * (1 + 0.03 * i / 12)                       # نمو سنوي ~3%
    season = 1 + 0.55 * math.sin(2 * math.pi * (month - 3) / 12)  # ذروة في الصيف
    value = base * season * random.uniform(0.95, 1.05)
    if (year, month) == (2025, 6):
        value *= 1.40                                       # ارتفاع غير معتاد قديم
    if (year, month) == (2026, 9):
        value *= 1.45                                       # ارتفاع غير معتاد حديث
    rows.append((f"{year}-{month:02d}-01", round(value, 1)))
    month += 1
    if month == 13:
        month, year = 1, year + 1

with open("sample_data.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "kwh"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows")
