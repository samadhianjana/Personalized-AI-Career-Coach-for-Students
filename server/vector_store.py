import pandas as pd
import numpy as np
import faiss
import json
import os
import kagglehub
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List
from datasets import load_dataset

# Hugging Face Datasets
job_descriptions = load_dataset("batuhanmtl/job-skill-set")["train"]
required_skillsets = load_dataset("aicinema69/Resume-Job")["train"]
salaries_skillsets = load_dataset("will4381/job-posting-classification")["train"]

# KaggleHub Datasets — returns file path, so load them manually
kaggle_job_path = kagglehub.dataset_download("arshkon/linkedin-job-postings")
kaggle_coursera_path = kagglehub.dataset_download("anusreemohanan/coursera-course-details")
kaggle_combined_path = kagglehub.dataset_download("kararhaitham/courses")

# Load CSVs/JSONs
linkedin_jobs = pd.read_csv(os.path.join(kaggle_job_path, "postings.csv"))
coursera_courses = pd.read_csv(os.path.join(kaggle_coursera_path, "coursera.csv"), encoding='latin-1')
combined_courses = pd.read_json(os.path.join(kaggle_combined_path, "combined_dataset.json"))

chunks = []

# Process batuhanmtl/job-skill-set
for row in job_descriptions:
    title = row.get("job_title", "")
    desc = row.get("job_description", "")
    skills = ", ".join(row.get("job_skill_set", [])) if row.get("job_skill_set") else ""
    combined = f"{title}. {desc}. Required skills: {skills}."
    chunks.append({
        "text": combined.strip(),
        "metadata": {"type": "job", "source": "huggingface_batuhanmtl", "title": title}
    })

# Process aicinema69/Resume-Job
for row in required_skillsets:
    title = row.get("job_title", "")
    skills = row.get("skills_required", "")
    desc = row.get("description", "")
    combined = f"{title}. Description: {desc}. Required Skills: {skills}."
    chunks.append({
        "text": combined.strip(),
        "metadata": {"type": "job", "source": "huggingface_aicinema69", "title": title}
    })

# Process will4381/job-posting-classification
for row in salaries_skillsets:
    title = row.get("title", "")
    category = row.get("category", "")
    desc = row.get("original_description", "")
    combined = f"{title}. {desc}. Category: {category}."
    chunks.append({
        "text": combined.strip(),
        "metadata": {"type": "job", "source": "huggingface_will4381", "title": title}
    })

# Process Kaggle: arshkon/linkedin-job-postings
for _, row in linkedin_jobs.iterrows(): # Changed job_listings to linkedin_jobs
    title = row.get("Job Title", "")
    desc = row.get("Description", "")
    salary = row.get("Salary Estimate", "")
    combined = f"{title}. {desc}. Salary: {salary}."
    chunks.append({
        "text": combined.strip(),
        "metadata": {"type": "job", "source": "kaggle_linkedin", "title": title}
    })

# Process Kaggle: anusreemohanan/coursera-course-details
for _, row in coursera_courses.iterrows():
    name = row.get("Course Name", "")
    desc = row.get("Course Description", "")
    level = row.get("Level", "")
    combined = f"{name}. Description: {desc}. Level: {level}."
    chunks.append({
        "text": combined.strip(),
        "metadata": {"type": "course", "source": "kaggle_coursera", "title": name}
    })

# Process Kaggle: kararhaitham/courses (combined)
for _, row in combined_courses.iterrows():
    name = row.get("course_title", "")
    desc = row.get("description", "")
    platform = row.get("platform", "")
    combined = f"{name}. {desc}. Platform: {platform}."
    chunks.append({
        "text": combined.strip(),
        "metadata": {"type": "course", "source": "kaggle_combined_courses", "title": name}
    })


model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & good enough for most tasks

texts = [chunk['text'] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)

# Create a FAISS index
dim = embeddings.shape[1]  # The dimension of the embeddings
index = faiss.IndexFlatL2(dim)  # Use L2 distance (Euclidean distance)

# Convert embeddings to a numpy array (FAISS requires this format)
embeddings_np = np.array(embeddings).astype(np.float32)

# Add embeddings to the FAISS index
index.add(embeddings_np)

# Save the FAISS index for later use
faiss.write_index(index, "career_coach_index.index")

