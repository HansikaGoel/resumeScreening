import re
from pathlib import Path
import pandas as pd
from services.resume_parser import extract_resume
from services.feature_extractor import extract_all_candidate_features


def process_bulk_resumes(folder_path, model=None):
    folder = Path(folder_path)

    results = []

    supported_formats = [
        ".pdf", ".docx", ".doc",
        ".txt", ".png", ".jpg", ".jpeg",
        ".tiff", ".bmp"
    ]

    if not folder.exists():
        return results

    # Find files in folder (and subfolders if structured)
    files = [
        file for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in supported_formats
    ]

    files.sort(key=lambda f: f.name)

    for file in files:
        clean_name = re.sub(r"^\d+_", "", file.name)
        try:
            text = extract_resume(str(file))
            features = extract_all_candidate_features(text)

            prediction = None
            probability = None
            shortlisted = "N/A"

            if model is not None:
                candidate_df = pd.DataFrame([{
                    "years_experience": features["years_experience"],
                    "skills_match_score": features["skills_match_score"],
                    "education_level": features["education_level"],
                    "project_count": features["project_count"],
                    "resume_length": features["resume_length"],
                    "github_activity": features["github_activity"]
                }])
                try:
                    pred = int(model.predict(candidate_df)[0])
                    prob = float(model.predict_proba(candidate_df)[0][1])
                    prediction = pred
                    probability = prob
                    shortlisted = "Yes" if pred == 1 else "No"
                except Exception:
                    pass

            results.append({
                "filename": file.name,
                "clean_filename": clean_name,
                "extracted_text": text,
                "features": features,
                "prediction": prediction,
                "probability": probability,
                "shortlisted": shortlisted,
                "status": "Success"
            })

        except Exception as e:
            results.append({
                "filename": file.name,
                "clean_filename": clean_name,
                "extracted_text": "",
                "features": {},
                "prediction": None,
                "probability": None,
                "shortlisted": "Error",
                "status": "Failed",
                "error": str(e)
            })

    return results
