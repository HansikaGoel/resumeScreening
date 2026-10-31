import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.resume_parser import extract_resume
from services.feature_extractor import extract_all_candidate_features

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    sample_file = BASE_DIR / "uploads" / "HansikaGoel.pdf"
    if sample_file.exists():
        file_path = str(sample_file)
    else:
        file_path = input("Enter resume file path: ")

print(f"\nProcessing: {file_path}")
try:
    text = extract_resume(file_path)
    print("\n--- Extracted Resume Text (first 500 chars) ---\n")
    print(text[:500] + ("..." if len(text) > 500 else ""))

    features = extract_all_candidate_features(text)
    print("\n--- Extracted Candidate Features ---")
    for k, v in features.items():
        print(f"  {k}: {v}")

except Exception as e:
    print("Error:", e)
