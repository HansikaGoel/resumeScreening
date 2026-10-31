from pathlib import Path
import sys
import joblib

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.bulk_processor import process_bulk_resumes

uploads_folder = BASE_DIR / "uploads"
model_path = BASE_DIR / "models" / "resume_model.pkl"

model = joblib.load(model_path) if model_path.exists() else None

results = process_bulk_resumes(str(uploads_folder), model=model)

print("\n==========================================")
print("     BULK RESUME PROCESSING RESULTS")
print("==========================================\n")

for result in results:
    print(f"File: {result['clean_filename']}")
    print(f"Status: {result['status']}")

    if result["status"] == "Success":
        feats = result["features"]
        print(f"Candidate: {feats.get('name', 'N/A')}")
        print(f"Education: {feats.get('education_level', 'N/A')} | Exp: {feats.get('years_experience', 0)} yrs")
        print(f"Skills Score: {feats.get('skills_match_score', 0)}% | Projects: {feats.get('project_count', 0)}")
        print(f"GitHub Activity: {feats.get('github_activity', 0)} | Word Count: {feats.get('resume_length', 0)}")
        print(f"Shortlist Prediction: {result.get('shortlisted', 'N/A')} (Prob: {result.get('probability', 0):.2%})")
    else:
        print(f"Error: {result['error']}")

    print("-" * 42)

total = len(results)
successful = sum(r["status"] == "Success" for r in results)
shortlisted = sum(r.get("shortlisted") == "Yes" for r in results)

print(f"\nTotal Resumes: {total}")
print(f"Successful: {successful} | Failed: {total - successful}")
print(f"Shortlisted: {shortlisted} / {successful}\n")
