from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import faiss
import pickle
import json
import re
import numpy as np
import google.generativeai as genai
import fitz  # PyMuPDF
import docx
import os

app = Flask(__name__)
# ========== Gemini API Key ==========
genai.configure(api_key="AIzaSyBi9ZKoZQx5yLmLyTtd7UV3XnYb85kaN84")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400
    
# ========== 2. Structured Output Using Gemini ==========

def get_structured_profile(text):
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
    You are a helpful assistant. Read the resume or transcript text below and extract the following fields in valid JSON format:
    - name (string)
    - degree (list of strings)
    - GPA (float, only the value obtained by the student — not the scale)
    - skills (list of strings)
    - interests (list of strings)
    Resume/Transcript Text:
    \"\"\"
    {text}
    \"\"\"
    Only return JSON object. Do not Do not explain anything.
    """
    response = model.generate_content(prompt)
    return response.text

# Helper to phrase lists nicely
def list_to_phrase(items):
    if len(items) == 0:
        return ""
    elif len(items) == 1:
        return items[0]
    else:
        return ", ".join(items[:-1]) + " and " + items[-1]

# Load index
index = faiss.read_index("career_coach_index.index")

# Load associated metadata for search results
with open("career_coach_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def search_vector(query: str, top_k: int = 5):
    # Generate the embedding for the query
    query_embedding = model.encode([query], normalize_embeddings=True)

    # Perform the search in the FAISS index
    query_embedding = np.array(query_embedding).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the closest matching chunks based on indices
    results = []
    for i, idx in enumerate(indices[0]):
        result = chunks[idx]
        result["distance"] = distances[0][i]  # Include the distance (similarity score)
        results.append(result)

    return results

def suggest_career_path_with_gemini(prepared_docs, student_profile_json):
    """
    Suggests career paths, required skills, and courses using Gemini and prepared documents.

    Args:
        prepared_docs (list): List of tuples containing (document_text, distance).
        student_profile_json (str): JSON string representing the student's profile.

    Returns:
        dict: A dictionary with keys 'career_paths', 'required_skills', and 'recommended_courses'.
    """

    # 1. Prepare context for Gemini
    context = "\n".join([doc for doc, _ in prepared_docs])
    context += f"\n\nStudent Profile:\n{student_profile_json}\n"
    context += "Databases:\nlinkedin_jobs, coursera_courses, combined_courses\n"

    # 2. Construct the prompt for a more conversational response
    prompt = f"""
You are a personalized AI career coach. Based on the student's profile and job description provided, suggest suitable career paths, required skills, and recommend courses. Present the results in the following format:
Context:
{context}

Hi, {{name}}, I'm your personalized AI career coach.
Based on your profile and the job description you provided, here's what I recommend:

These are the career paths I suggest for you: {{career_paths}}.

These are the required skills you should focus on: {{required_skills}}. You already have a strong foundation in areas like project management, time management, and communication, so focusing on the specifics mentioned in the job description will make you a stronger candidate.

These are the courses I would recommend to you:
{{courses_list}}

Format the career paths, required skills, and courses in a clean, readable format without extra markdown or quotes.
"""

    # 3. Generate response using Gemini
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    output_text = response.text.strip()

    # 4. Use `re` to clean up and format the response
    # This regex will clean up unnecessary markdown and add the personalized format
    output_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", output_text.strip())  # Remove any code block
    output_text = re.sub(r"career paths", "career paths", output_text, flags=re.IGNORECASE)  # Ensure consistency in capitalization

    # 5. Format the final response with re.sub to insert the correct labels in the response
    output_text = re.sub(
        r"(career paths.*?:)(.*?)\.",
        r"These are the career paths I suggest for you: \2.",
        output_text
    )

    output_text = re.sub(
        r"(required skills.*?:)(.*?)\.",
        r"These are the required skills you should focus on: \2.",
        output_text
    )

    output_text = re.sub(
        r"(recommended courses.*?:)(.*?)\.",
        r"These are the courses I would recommend to you: \2.",
        output_text
    )

    # 6. Return the cleaned and formatted response
    return output_text


# ----------- API Route -----------

@app.route("/analyze", methods=["POST"])
def analyze_document():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Extract text
    raw_text = extract_text(file_path)
    profile_json = get_structured_profile(raw_text)

    profile_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", profile_json.strip())

    student_profile_json = profile_json

    # Parse the JSON into a Python dictionary
    try:
        profile = json.loads(student_profile_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

    # Validate required fields
    required_fields = ["degree", "GPA", "interests", "skills"]
    missing = [field for field in required_fields if field not in profile]

    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    # Smart Query Construction

    degree = profile.get("degree", "unknown degree")
    gpa = profile.get("GPA", 0.0)
    interests = profile.get("interests", [])
    skills = profile.get("skills", [])


    # Analyze strength level based on GPA
    if gpa >= 3.5:
        performance = "strong academic performance"
    elif 3.0 <= gpa < 3.5:
        performance = "average academic performance"
    else:
        performance = "needs improvement in academic performance"

    # Build a flexible natural language query
    query = (
        f"Suggest high-potential career opportunities and emerging job roles for a student "
        f"with a {degree}, {performance}, skilled in {list_to_phrase(skills)}, and interested in {list_to_phrase(interests)}. "
        f"Also recommend relevant certifications or learning paths to enhance employability in these fields."
    )

        # 1. Call Naveen's search_vector() function
    retrieved_results = search_vector(query, top_k=5)

    # 2. Check if results are retrieved
    if not retrieved_results:
        print("No documents retrieved! Please check the query.")
    else:
        print(f"Retrieved {len(retrieved_results)} documents.\n")
    
    prepared_docs = []

    for result in retrieved_results:
        title = result.get('metadata', {}).get('title', 'No Title')
        text = result.get('text', 'No Text Available')
        distance = result.get('distance', None)

        # Relaxed filter
        if distance is not None and distance > 0.8:  # Not very strict now
            continue

        combined_text = f"Title: {title}\nContent: {text}\nSimilarity Score: {distance}\n"
        prepared_docs.append((combined_text, distance))  # Save both text + distance for sorting later

        # Sort by distance (smallest first = most similar)
        prepared_docs = sorted(prepared_docs, key=lambda x: x[1])

        # Take only top 5
        prepared_docs = prepared_docs[:5]

        # Print nicely
        for idx, (doc, distance) in enumerate(prepared_docs, start=1):
            print(f"Document {idx}:\n{doc}\n{'-'*80}\n")

        # 4. Generate career advice using Gemini
    career_advice = suggest_career_path_with_gemini(prepared_docs, student_profile_json)

    # 5. Display results
    if career_advice:
        print(career_advice)

    return jsonify({
        "query": query,
        "recommendations": recommendations
    })

# ----------- Main -----------

if __name__ == "__main__":
    app.run(debug=True)