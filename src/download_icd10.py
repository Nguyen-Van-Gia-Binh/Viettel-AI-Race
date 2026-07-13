"""Tải toàn bộ danh sách mã ICD-10-CM (NLM Clinical Tables API, public, no auth)
về local để dùng offline cho candidate mapping. Chỉ chạy 1 lần khi chuẩn bị data,
KHÔNG gọi API này lúc inference.
"""
import csv
import os
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "icd10cm_full.csv")


def fetch(terms, max_list=500):
    r = requests.get(API, params={"sf": "code,name", "terms": terms, "maxList": max_list}, timeout=20)
    r.raise_for_status()
    data = r.json()
    total = data[0]
    pairs = data[3] or []
    return total, pairs


def main():
    all_codes = {}  # code -> name

    letters = list(string.ascii_uppercase)
    # Level 1: per-letter count
    letter_counts = {}
    for letter in letters:
        try:
            total, pairs = fetch(letter)
        except Exception as e:
            print(f"[WARN] letter {letter} failed: {e}")
            continue
        letter_counts[letter] = total
        if total == 0:
            continue
        if total <= 500:
            for code, name in pairs:
                all_codes[code] = name
        # else refine below
        time.sleep(0.05)

    # Level 2: refine letters with >500 into letter+2digit prefixes
    to_refine = [l for l, c in letter_counts.items() if c > 500]
    print(f"Letters needing refinement: {to_refine}")

    def fetch_prefix(prefix):
        try:
            total, pairs = fetch(prefix)
            return prefix, total, pairs
        except Exception as e:
            return prefix, -1, []

    tasks = []
    for letter in to_refine:
        for d in range(0, 100):
            tasks.append(f"{letter}{d:02d}")

    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = [ex.submit(fetch_prefix, p) for p in tasks]
        done = 0
        for fut in as_completed(futures):
            prefix, total, pairs = fut.result()
            done += 1
            if total == -1:
                print(f"[WARN] prefix {prefix} failed")
                continue
            if total > 500:
                print(f"[WARN] prefix {prefix} still has {total} > 500, need level-3 refinement")
            for code, name in pairs:
                all_codes[code] = name
            if done % 200 == 0:
                print(f"Progress: {done}/{len(tasks)} prefixes, {len(all_codes)} codes so far")

    print(f"Total unique codes collected: {len(all_codes)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "description_en"])
        for code in sorted(all_codes):
            w.writerow([code, all_codes[code]])

    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
