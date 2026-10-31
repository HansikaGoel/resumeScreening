import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "ai_resume_screening.csv"


class DatasetBenchmark:
    """
    Computes and caches conditional benchmarks from ai_resume_screening.csv.
    When a resume is missing any feature, it supplies an intelligent conditional average
    based on the other features detected in the resume.
    """
    _instance = None
    _stats = None

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        if cls._stats is not None:
            return cls._stats

        # Fallback defaults based on dataset if CSV cannot be loaded
        default_stats = {
            "overall": {
                "years_experience": 7.5,
                "skills_match_score": 73.7,
                "project_count": 11,
                "resume_length": 573,
                "github_activity": 325,
                "education_level": "Bachelors"
            },
            "by_exp": {
                "0-1": {"github_activity": 126, "project_count": 6, "resume_length": 367, "skills_match_score": 60.1},
                "2-4": {"github_activity": 194, "project_count": 7, "resume_length": 439, "skills_match_score": 65.4},
                "5-8": {"github_activity": 296, "project_count": 10, "resume_length": 542, "skills_match_score": 72.1},
                "9-12": {"github_activity": 413, "project_count": 13, "resume_length": 665, "skills_match_score": 79.4},
                "13+": {"github_activity": 511, "project_count": 15, "resume_length": 760, "skills_match_score": 85.5}
            },
            "by_proj": {
                "0-5": {"years_experience": 2.5, "skills_match_score": 62.0, "github_activity": 150, "resume_length": 426},
                "6-10": {"years_experience": 5.6, "skills_match_score": 68.5, "github_activity": 260, "resume_length": 516},
                "11-15": {"years_experience": 9.6, "skills_match_score": 77.0, "github_activity": 410, "resume_length": 634},
                "16+": {"years_experience": 12.2, "skills_match_score": 84.0, "github_activity": 530, "resume_length": 710}
            }
        }

        if DATASET_PATH.exists():
            try:
                df = pd.read_csv(DATASET_PATH)

                # Overall means
                overall = {
                    "years_experience": round(float(df["years_experience"].mean()), 1),
                    "skills_match_score": round(float(df["skills_match_score"].mean()), 1),
                    "project_count": int(round(df["project_count"].mean())),
                    "resume_length": int(round(df["resume_length"].mean())),
                    "github_activity": int(round(df["github_activity"].mean())),
                    "education_level": df["education_level"].mode()[0] if "education_level" in df else "Bachelors"
                }

                # Conditional on experience tiers
                exp_bins = [-1, 1, 4, 8, 12, 100]
                exp_labels = ["0-1", "2-4", "5-8", "9-12", "13+"]
                df["exp_tier"] = pd.cut(df["years_experience"], bins=exp_bins, labels=exp_labels)
                by_exp = df.groupby("exp_tier", observed=False)[
                    ["skills_match_score", "project_count", "resume_length", "github_activity"]
                ].mean().round(1).to_dict(orient="index")

                # Conditional on project count tiers
                proj_bins = [-1, 5, 10, 15, 100]
                proj_labels = ["0-5", "6-10", "11-15", "16+"]
                df["proj_tier"] = pd.cut(df["project_count"], bins=proj_bins, labels=proj_labels)
                by_proj = df.groupby("proj_tier", observed=False)[
                    ["years_experience", "skills_match_score", "github_activity", "resume_length"]
                ].mean().round(1).to_dict(orient="index")

                cls._stats = {
                    "overall": overall,
                    "by_exp": by_exp,
                    "by_proj": by_proj
                }
                return cls._stats
            except Exception:
                pass

        cls._stats = default_stats
        return cls._stats

    @classmethod
    def get_exp_tier(cls, exp: float) -> str:
        if exp <= 1:
            return "0-1"
        elif exp <= 4:
            return "2-4"
        elif exp <= 8:
            return "5-8"
        elif exp <= 12:
            return "9-12"
        return "13+"

    @classmethod
    def get_proj_tier(cls, proj: int) -> str:
        if proj <= 5:
            return "0-5"
        elif proj <= 10:
            return "6-10"
        elif proj <= 15:
            return "11-15"
        return "16+"


def extract_candidate_name(text: str) -> str:
    """Extract candidate name from the top lines of resume text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Unknown Candidate"

    ignore_keywords = [
        "resume", "curriculum", "vitae", "cv", "page", "phone",
        "email", "github", "linkedin", "profile", "contact"
    ]

    for line in lines[:5]:
        line_clean = re.sub(r"^[^\w\s]+", "", line).strip()
        lower = line_clean.lower()
        if any(keyword in lower for keyword in ignore_keywords):
            continue
        words = line_clean.split()
        if 1 <= len(words) <= 4 and not re.search(r"[\d@:/]", line_clean):
            if len(line_clean) <= 40:
                return line_clean

    return "Candidate"


def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract email and phone."""
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match.group() if email_match else ""

    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}", text)
    phone = phone_match.group().strip() if phone_match else ""

    return {"email": email, "phone": phone}


def extract_education_level(text: str) -> Tuple[str, bool]:
    """
    Returns (education_level, is_explicitly_detected)
    """
    text_lower = text.lower()

    if re.search(r"\b(ph\.?d|doctorate|doctor of philosophy|post\s*doc(?:toral)?)\b", text_lower):
        return ("PhD", True)

    if re.search(r"\b(master(?:'s)?|m\.?tech|m\.?s\.?|m\.?sc|mba|mca|m\.?e\.?|post\s*graduate)\b", text_lower):
        return ("Masters", True)

    if re.search(r"\b(bachelor(?:'s)?|b\.?tech|b\.?e\.?|b\.?s\.?|b\.?sc|bca|b\.?com|b\.?a\.?|undergraduate)\b", text_lower):
        return ("Bachelors", True)

    if re.search(r"\b(high\s*school|secondary\s*school|higher\s*secondary|intermediate|12th|10th)\b", text_lower):
        return ("High School", True)

    # Missing from resume: will be imputed from dataset mode
    return ("Bachelors", False)


def extract_years_experience(text: str) -> Tuple[int, bool]:
    """
    Returns (years_experience, is_explicitly_detected)
    """
    text_lower = text.lower()

    # Explicit mentions like '3+ years of experience' or '5 yrs exp'
    explicit_match = re.search(
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:professional\s+)?experience",
        text_lower
    )
    if explicit_match:
        try:
            val = float(explicit_match.group(1))
            return (int(min(max(val, 0), 50)), True)
        except ValueError:
            pass

    # Check future graduation (college student / fresher)
    future_grad = re.search(r"(?:202[5-9]|203[0-9])", text)
    if future_grad:
        internships = len(re.findall(r"\bintern(?:ship)?\b", text_lower))
        return (1 if internships >= 2 else 0, True)

    # Look for year ranges (e.g., 2019 - 2023)
    year_matches = [int(y) for y in re.findall(r"\b(20\d\d|19\d\d)\b", text)]
    valid_years = [y for y in year_matches if 1990 <= y <= 2026]

    if len(valid_years) >= 2:
        diff = max(valid_years) - min(valid_years)
        return (int(min(max(diff, 0), 30)), True)

    return (0, False)


def extract_project_count(text: str) -> Tuple[int, bool]:
    """
    Returns (project_count, is_explicitly_detected)
    """
    text_lower = text.lower()

    # Numbered projects: '1. Portfolio Website'
    numbered = len(re.findall(r"(?m)^\s*(?:\d+[\.\)]|project\s*\d+:)\s+\w+", text))
    if numbered > 0:
        return (int(min(numbered, 100)), True)

    # Bullets inside projects section
    in_project_section = False
    bullet_count = 0
    for line in text.splitlines():
        line_clean = line.strip().lower()
        if "project" in line_clean and len(line_clean) < 30:
            in_project_section = True
            continue
        if in_project_section:
            if any(heading in line_clean for heading in ["education", "experience", "skills", "certificate", "achievement"]):
                break
            if line.strip().startswith(("•", "-", "*", "–")):
                bullet_count += 1

    if bullet_count > 0:
        return (int(min(max(bullet_count // 2, 1), 30)), True)

    # Mentions of action verbs
    action_matches = len(re.findall(r"\b(?:developed|built|created|implemented|designed)\s+(?:a|an|the)?\s+\w+", text_lower))
    if action_matches >= 2:
        return (int(min(action_matches, 20)), True)

    return (0, False)


def extract_resume_length(text: str) -> Tuple[int, bool]:
    """Returns (word_count, is_adequate_length)."""
    words = len(text.split())
    if words >= 80:
        return (int(min(max(words, 50), 2000)), True)
    return (words, False)


def extract_github_activity(text: str) -> Tuple[int, bool]:
    """
    Returns (github_activity, is_explicitly_detected)
    """
    text_lower = text.lower()
    has_git = bool(re.search(r"\b(github|gitlab|bitbucket|git)\b", text_lower))
    has_coding = bool(re.search(r"\b(codechef|leetcode|hackerrank|codeforces)\b", text_lower))

    if has_git or has_coding:
        score = 150
        count_matches = re.findall(
            r"(\d+)\+?\s*(?:dsa|problems?|contributions?|commits?|repos?|repositories|stars?|views?)",
            text_lower
        )
        for count_str in count_matches:
            try:
                score += min(int(count_str), 500)
            except ValueError:
                pass

        if "open source" in text_lower or "contributor" in text_lower:
            score += 75
        if "ambassador" in text_lower or "maintainer" in text_lower:
            score += 50
        return (int(min(max(score, 0), 2000)), True)

    return (0, False)


def extract_skills(text: str) -> Tuple[float, List[str], bool]:
    """
    Returns (skills_match_score, detected_skills, is_adequate_skills)
    """
    text_lower = text.lower()

    technical_skills = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "sql", "html", "html5",
        "css", "css3", "bash", "shell",
        "react", "react.js", "angular", "vue", "vue.js", "node", "node.js", "express",
        "express.js", "django", "flask", "fastapi", "spring", "spring boot", "next.js",
        "redux", "tailwind", "bootstrap",
        "mongodb", "postgresql", "postgres", "mysql", "sqlite", "redis", "oracle",
        "cassandra", "dynamodb", "firebase",
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github",
        "linux", "ci/cd", "terraform", "nginx",
        "machine learning", "deep learning", "nlp", "computer vision", "pytorch",
        "tensorflow", "pandas", "numpy", "scikit-learn", "sklearn", "tableau", "power bi",
        "dsa", "data structures", "algorithms", "oops", "object-oriented", "rest", "rest api",
        "microservices", "system design", "copilot", "chatgpt", "gemini", "claude"
    ]

    detected = []
    if re.search(r"\b(?:c\s+programming|c\s+language|\blanguages?:.*?\bc\b)", text_lower):
        detected.append("c")
    if re.search(r"\b(?:r\s+programming|r\s+language|r\s+script|\blanguages?:.*?\br\b)", text_lower):
        detected.append("r")

    for skill in technical_skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            detected.append(skill)

    unique_skills = []
    seen = set()
    for s in detected:
        canonical = s.replace(".js", "").replace("5", "").replace("3", "").strip()
        if canonical not in seen:
            seen.add(canonical)
            unique_skills.append(s)

    count = len(unique_skills)
    if count >= 3:
        score = min(98.0, max(25.0, 32.0 + count * 3.6))
        return (round(score, 1), unique_skills, True)

    score = 32.0 + count * 3.6
    return (round(score, 1), unique_skills, False)


def extract_all_candidate_features(text: str) -> Dict[str, Any]:
    """
    Extracts all candidate features from resume text.
    If ANY detail is missing or not detected in the resume, it calculates and provides
    an intelligent conditional average derived from the provided dataset (ai_resume_screening.csv)
    and the other detected candidate details.
    """
    stats = DatasetBenchmark.get_stats()
    imputed_fields = {}

    name = extract_candidate_name(text)
    contact = extract_contact_info(text)

    # 1. Raw Extractions
    edu, edu_found = extract_education_level(text)
    exp, exp_found = extract_years_experience(text)
    proj, proj_found = extract_project_count(text)
    rlen, rlen_found = extract_resume_length(text)
    gh, gh_found = extract_github_activity(text)
    skills_score, skills_detected, skills_found = extract_skills(text)

    # 2. Imputation of Missing Details based on the Database & Other Details

    # If Experience was not detected:
    if not exp_found:
        if proj_found:
            p_tier = DatasetBenchmark.get_proj_tier(proj)
            imputed_exp = stats["by_proj"][p_tier]["years_experience"]
            exp = int(round(imputed_exp))
            imputed_fields["years_experience"] = f"Imputed: {exp} yrs (dataset avg for candidates with {proj} projects)"
        else:
            exp = int(round(stats["overall"]["years_experience"]))
            imputed_fields["years_experience"] = f"Imputed: {exp} yrs (overall dataset average)"

    exp_tier = DatasetBenchmark.get_exp_tier(exp)
    tier_stats = stats["by_exp"].get(exp_tier, stats["overall"])

    # If Education was not detected:
    if not edu_found:
        edu = stats["overall"]["education_level"]
        imputed_fields["education_level"] = f"Imputed: {edu} (most frequent degree in dataset)"

    # If Project Count was not detected:
    if not proj_found:
        proj = int(round(tier_stats.get("project_count", stats["overall"]["project_count"])))
        imputed_fields["project_count"] = f"Imputed: {proj} projects (dataset avg for {exp} yrs experience)"

    # If GitHub Activity was not detected:
    if not gh_found:
        gh = int(round(tier_stats.get("github_activity", stats["overall"]["github_activity"])))
        imputed_fields["github_activity"] = f"Imputed: {gh} (dataset avg activity for {exp} yrs experience)"

    # If Skills Match Score is too low/not detected:
    if not skills_found:
        tier_skill_avg = tier_stats.get("skills_match_score", stats["overall"]["skills_match_score"])
        skills_score = round(float(tier_skill_avg), 1)
        imputed_fields["skills_match_score"] = f"Imputed: {skills_score}% (dataset avg for {exp} yrs experience)"

    # If Resume Length was inadequate/missing:
    if not rlen_found:
        rlen = int(round(tier_stats.get("resume_length", stats["overall"]["resume_length"])))
        imputed_fields["resume_length"] = f"Imputed: {rlen} words (dataset avg for {exp} yrs experience)"

    return {
        "name": name,
        "email": contact["email"],
        "phone": contact["phone"],
        "education_level": edu,
        "years_experience": exp,
        "project_count": proj,
        "resume_length": rlen,
        "github_activity": gh,
        "skills_match_score": skills_score,
        "skills_detected": skills_detected,
        "imputed_fields": imputed_fields
    }
