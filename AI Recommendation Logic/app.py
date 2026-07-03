"""AI Recommendation Logic - Project 3.

DecodeLabs Internship Program | Batch: 2026

This script implements a content-based recommendation engine for job roles.
It ingests three user skills, maps them to the same vocabulary as a curated
role dataset using TF-IDF, scores the roles with cosine similarity, and returns
Top-N recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_TITLE = "AI Recommendation Logic - Project 3"
OUTPUT_DIR = Path(__file__).with_name("artifacts")
TOP_N = 3

SKILL_ALIASES = {
    "cloud computing": "cloud",
    "automation": "devops",
    "ci cd": "devops",
    "continuous integration": "devops",
    "continuous delivery": "devops",
    "frontend": "frontend development",
    "backend": "backend development",
    "machine learning": "machine learning",
    "deep learning": "deep learning",
}

JOB_ROLES_DATA = {
    "Job_Title": [
        "Data Scientist",
        "Machine Learning Engineer",
        "Cloud Architect",
        "Web Developer",
        "DevOps Engineer",
        "Data Analyst",
        "Full Stack Developer",
        "Python Developer",
        "Frontend Developer",
        "Backend Developer",
        "Security Engineer",
        "Mobile Developer",
    ],
    "Skills": [
        "Python SQL Machine Learning Statistics Deep Learning Pandas",
        "Python TensorFlow PyTorch Docker Kubernetes MLflow",
        "AWS Azure GCP Terraform Kubernetes Python DevOps",
        "HTML CSS JavaScript React Node.js MongoDB Express",
        "Linux Docker Kubernetes AWS Jenkins CI/CD Python Shell",
        "SQL Excel Tableau Python Pandas Statistics Power BI",
        "JavaScript React Node.js MongoDB Express HTML CSS",
        "Python Django Flask REST API SQL Git",
        "HTML CSS JavaScript React TypeScript Webpack",
        "Node.js Python Java Spring Boot SQL REST API",
        "Network Security Python Firewalls SIEM Linux AWS",
        "iOS Android Swift Kotlin React Native Java",
    ],
    "Category": [
        "Data Science",
        "Machine Learning",
        "Cloud",
        "Web Development",
        "DevOps",
        "Data Science",
        "Web Development",
        "Development",
        "Web Development",
        "Development",
        "Security",
        "Mobile Development",
    ],
}


@dataclass(slots=True)
class RecommendationResult:
    """Container for ranked recommendation output."""

    user_skills: list[str]
    ranked_jobs: pd.DataFrame
    top_n: pd.DataFrame
    vectorizer: TfidfVectorizer


def build_job_dataframe() -> pd.DataFrame:
    """Create the job-role dataset."""

    return pd.DataFrame(JOB_ROLES_DATA)


def collect_user_skills() -> list[str]:
    """Collect at least three user skills from the command line.

    The script accepts comma-separated input and falls back to a formal default
    profile if no usable input is provided, so the program can still execute in
    non-interactive environments.
    """

    default_skills = ["Python", "Cloud Computing", "Automation"]
    print("Enter your top 3 skills or interests, separated by commas.")
    print("Example: Python, Cloud Computing, Automation")

    try:
        raw_input = input("Your skills: ").strip()
    except EOFError:
        raw_input = ""

    if not raw_input:
        print("No input detected. Using default profile for demonstration.")
        return default_skills

    skills = [skill.strip() for skill in raw_input.split(",") if skill.strip()]

    while len(skills) < 3:
        try:
            additional_input = input("Add more skills: ").strip()
        except EOFError:
            additional_input = ""

        if not additional_input:
            break

        skills.extend(skill.strip() for skill in additional_input.split(",") if skill.strip())

    unique_skills = []
    for skill in skills:
        if skill.lower() not in {item.lower() for item in unique_skills}:
            unique_skills.append(skill)

    return unique_skills[:3] if len(unique_skills) >= 3 else default_skills


def build_user_profile(user_skills: Iterable[str]) -> str:
    """Convert user skills into a normalized profile string."""

    normalized_skills: list[str] = []

    for skill in user_skills:
        cleaned_skill = skill.lower().strip()
        canonical_skill = SKILL_ALIASES.get(cleaned_skill, cleaned_skill)
        normalized_skills.extend(canonical_skill.split())

    return " ".join(normalized_skills)


def score_recommendations(user_skills: list[str], jobs: pd.DataFrame) -> RecommendationResult:
    """Calculate cosine similarity between the user profile and all job roles."""

    user_profile = build_user_profile(user_skills)
    documents = jobs["Skills"].tolist() + [user_profile]

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    job_matrix = tfidf_matrix[:-1]
    user_matrix = tfidf_matrix[-1]

    similarities = cosine_similarity(user_matrix, job_matrix).flatten()

    ranked_jobs = jobs.copy()
    ranked_jobs["Similarity_Score"] = similarities
    ranked_jobs = ranked_jobs.sort_values("Similarity_Score", ascending=False).reset_index(drop=True)
    top_n = ranked_jobs.head(TOP_N).copy()

    return RecommendationResult(
        user_skills=user_skills,
        ranked_jobs=ranked_jobs,
        top_n=top_n,
        vectorizer=vectorizer,
    )


def render_report(result: RecommendationResult) -> None:
    """Print the recommendation results to the console."""

    print()
    print(APP_TITLE)
    print("=" * len(APP_TITLE))
    print("Dataset Shape: (12, 3)")
    print()
    print(f"Based on your skills: {', '.join(result.user_skills)}")
    print()
    print(f"Top {TOP_N} Recommended Job Roles:")
    print("-" * 60)

    for rank, (_, row) in enumerate(result.top_n.iterrows(), start=1):
        print(f"#{rank}: {row['Job_Title']}")
        print(f"   Category: {row['Category']}")
        print(f"   Similarity Score: {row['Similarity_Score'] * 100:.2f}%")
        print(f"   Key Skills: {row['Skills']}")
        print("-" * 40)

    print()
    print("DETAILED RECOMMENDATION REPORT")
    print("=" * 60)
    print(f"USER SKILLS: {', '.join(skill.upper() for skill in result.user_skills)}")
    print()

    for rank, (_, row) in enumerate(result.top_n.iterrows(), start=1):
        print(f"{rank}. {row['Job_Title']} ({row['Category']})")
        print(f"   Match Score: {row['Similarity_Score'] * 100:.1f}%")
        user_skills_lower = {skill.lower() for skill in result.user_skills}
        job_skills_lower = set(row["Skills"].lower().split())
        matching_skills = sorted(skill for skill in user_skills_lower if skill in job_skills_lower)
        if matching_skills:
            print(f"   Matching Skills: {', '.join(matching_skills)}")
        print(f"   Required Skills: {row['Skills']}")
        print()


def save_chart(ranked_jobs: pd.DataFrame, user_skills: list[str]) -> Path:
    """Save a bar chart for the top recommendations."""

    OUTPUT_DIR.mkdir(exist_ok=True)
    chart_path = OUTPUT_DIR / "tech_stack_recommendations.png"

    top_10 = ranked_jobs.head(10)
    scores = top_10["Similarity_Score"] * 100
    titles = top_10["Job_Title"]

    colors = ["#FF6B6B" if index == 0 else "#4ECDC4" if index < 3 else "#45B7D1" for index in range(len(titles))]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(titles)), scores, color=colors)
    plt.xlabel("Similarity Score (%)")
    plt.title(f"Top 10 Job Role Recommendations\nBased on Skills: {', '.join(user_skills[:3])}")
    plt.yticks(range(len(titles)), titles)

    for index, value in enumerate(scores):
        plt.text(value + 0.5, index, f"{value:.1f}%", va="center")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path


def main() -> None:
    """Run the recommendation workflow."""

    jobs = build_job_dataframe()
    user_skills = collect_user_skills()
    result = score_recommendations(user_skills, jobs)
    chart_path = save_chart(result.ranked_jobs, result.user_skills)

    print(f"Dataset Shape: {jobs.shape}")
    print("\nAvailable Job Roles:")
    print(jobs["Job_Title"].tolist())
    print()
    print("=" * 60)
    print("STEP 1: INGESTION - User Profile Input")
    print("=" * 60)
    print(f"User Profile Created: {build_user_profile(result.user_skills)}")
    print()
    print("=" * 60)
    print("STEP 2: SCORING - Calculating Similarity")
    print("=" * 60)
    print(f"Calculated similarity scores for {len(jobs)} job roles.")
    print(f"Max Score: {result.ranked_jobs['Similarity_Score'].max():.4f}")
    print(f"Min Score: {result.ranked_jobs['Similarity_Score'].min():.4f}")
    print()
    print("=" * 60)
    print("STEP 3: SORTING - Ranking Recommendations")
    print("=" * 60)
    print("Top recommendations based on similarity scores:")
    print()
    print("=" * 60)
    print("STEP 4: FILTERING - Top-N Recommendations")
    print("=" * 60)
    render_report(result)
    print("=" * 60)
    print("RECOMMENDATION COMPLETE")
    print("=" * 60)
    print(f"Saved recommendation chart: {chart_path}")


if __name__ == "__main__":
    main()
