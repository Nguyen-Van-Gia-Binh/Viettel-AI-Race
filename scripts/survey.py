"""Khảo sát nhanh 100 file test để nắm phân bố nội dung trước khi viết baseline."""
import re
import os
from collections import Counter

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input", "input")

def main():
    files = sorted(os.listdir(INPUT_DIR), key=lambda x: int(x.split(".")[0]))
    lengths_lines = []
    lengths_chars = []
    section_headers = Counter()
    all_text = []

    header_pattern = re.compile(r"^\s*\d*\.?\s*([A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zàảãáạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ ]{3,40})\s*$")

    for f in files:
        path = os.path.join(INPUT_DIR, f)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        all_text.append((f, text))
        lengths_chars.append(len(text))
        lines = text.split("\n")
        lengths_lines.append(len(lines))
        for line in lines:
            stripped = line.strip()
            if 3 < len(stripped) < 45 and stripped == stripped.strip():
                # Heuristic: dòng ngắn, không có dấu câu cuối, không phải bullet
                if not stripped.startswith("-") and not re.search(r"[.:][a-zA-Zà-ỹ]", stripped):
                    m = header_pattern.match(stripped)
                    if m:
                        section_headers[stripped.lower()] += 1

    print(f"Tổng số file: {len(files)}")
    print(f"Số dòng: min={min(lengths_lines)}, max={max(lengths_lines)}, avg={sum(lengths_lines)/len(lengths_lines):.1f}")
    print(f"Số ký tự: min={min(lengths_chars)}, max={max(lengths_chars)}, avg={sum(lengths_chars)/len(lengths_chars):.1f}")
    print()
    print("=== Top 40 section headers tiềm năng ===")
    for h, c in section_headers.most_common(40):
        print(f"{c:3d}  {h}")

    # Common abbreviation-like tokens (all caps or lowercase short codes)
    abbr_counter = Counter()
    for f, text in all_text:
        for tok in re.findall(r"\b[a-zA-Z]{2,6}\b", text):
            if tok.lower() in ("po","bid","prn","qid","qam","qhs","q6h","ecg","spo2","wbc","neut","lymph","ct","mri","hb","rbc","plt", "bpm", "mg", "ml", "kg","cm"):
                abbr_counter[tok.lower()] += 1
    print()
    print("=== Viết tắt phổ biến ===")
    for a, c in abbr_counter.most_common(30):
        print(f"{c:3d}  {a}")

if __name__ == "__main__":
    main()
