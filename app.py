from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


app = Flask(__name__)


# ============================================================
# ATS SKILL LIBRARY
# ============================================================
# Each skill has:
# - aliases: different ways it may appear in a resume/JD
# - category: used for score breakdown
# - weight: importance inside the skill score
#
# This keeps the project completely free/local and avoids
# requiring an external AI API.
# ============================================================

SKILLS = {
    # Core AI / LLM
    "Python": {
        "aliases": ["python"],
        "category": "Programming",
        "weight": 1.0,
    },
    "Generative AI": {
        "aliases": ["generative ai", "gen ai", "genai"],
        "category": "AI / LLM",
        "weight": 1.0,
    },
    "LLMs": {
        "aliases": ["llm", "llms", "large language model", "large language models"],
        "category": "AI / LLM",
        "weight": 1.0,
    },
    "RAG": {
        "aliases": [
            "rag",
            "retrieval-augmented generation",
            "retrieval augmented generation",
        ],
        "category": "AI / LLM",
        "weight": 1.2,
    },
    "AI Agents": {
        "aliases": ["ai agent", "ai agents", "agentic ai", "agentic"],
        "category": "AI / LLM",
        "weight": 1.0,
    },
    "Tool / Function Calling": {
        "aliases": [
            "tool/function calling",
            "tool calling",
            "function calling",
            "function-calling",
        ],
        "category": "AI / LLM",
        "weight": 0.9,
    },
    "Chatbots": {
        "aliases": ["chatbot", "chatbots", "conversational ai"],
        "category": "AI / LLM",
        "weight": 1.0,
    },
    "Embeddings": {
        "aliases": ["embedding", "embeddings"],
        "category": "AI / Retrieval",
        "weight": 1.0,
    },
    "Vector Databases": {
        "aliases": ["vector database", "vector databases", "vector db", "vector dbs"],
        "category": "AI / Retrieval",
        "weight": 1.0,
    },
    "Semantic Retrieval": {
        "aliases": ["semantic retrieval", "semantic search"],
        "category": "AI / Retrieval",
        "weight": 0.8,
    },

    # Frameworks / providers
    "LangChain": {
        "aliases": ["langchain"],
        "category": "AI Tools",
        "weight": 0.9,
    },
    "LangGraph": {
        "aliases": ["langgraph"],
        "category": "AI Tools",
        "weight": 0.8,
    },
    "LlamaIndex": {
        "aliases": ["llamaindex", "llama index"],
        "category": "AI Tools",
        "weight": 0.8,
    },
    "Gemini": {
        "aliases": ["gemini", "google gemini"],
        "category": "LLM APIs",
        "weight": 0.9,
    },
    "OpenAI": {
        "aliases": ["openai", "openai api", "openai apis"],
        "category": "LLM APIs",
        "weight": 0.7,
    },
    "Claude": {
        "aliases": ["claude", "anthropic", "anthropic api"],
        "category": "LLM APIs",
        "weight": 0.7,
    },

    # Retrieval providers
    "Pinecone": {
        "aliases": ["pinecone"],
        "category": "Vector Databases",
        "weight": 0.9,
    },
    "FAISS": {
        "aliases": ["faiss"],
        "category": "Vector Databases",
        "weight": 0.7,
    },
    "ChromaDB": {
        "aliases": ["chromadb", "chroma db", "chroma"],
        "category": "Vector Databases",
        "weight": 0.7,
    },
    "Weaviate": {
        "aliases": ["weaviate"],
        "category": "Vector Databases",
        "weight": 0.7,
    },
    "HuggingFace": {
        "aliases": ["huggingface", "hugging face"],
        "category": "AI Tools",
        "weight": 0.7,
    },

    # Engineering
    "Git / GitHub": {
        "aliases": ["git/github", "github", "git"],
        "category": "Engineering",
        "weight": 0.8,
    },
    "REST APIs": {
        "aliases": ["rest api", "rest apis", "restful api", "restful apis", "api development"],
        "category": "Engineering",
        "weight": 0.9,
    },
    "Flask": {
        "aliases": ["flask"],
        "category": "Backend",
        "weight": 0.7,
    },
    "Node.js": {
        "aliases": ["node.js", "nodejs", "node js"],
        "category": "Backend",
        "weight": 0.5,
    },
}


# Requirements that the JD explicitly treats as core expectations.
CORE_SKILLS = {
    "Python",
    "Generative AI",
    "LLMs",
    "RAG",
    "Chatbots",
}

# Skills described as "plus", "good to have", or similar.
PREFERRED_SKILLS = {
    "AI Agents",
    "Tool / Function Calling",
    "Embeddings",
    "Vector Databases",
    "LangChain",
    "LangGraph",
    "LlamaIndex",
    "Gemini",
    "OpenAI",
    "Claude",
    "Pinecone",
    "FAISS",
    "ChromaDB",
    "Weaviate",
    "Git / GitHub",
    "REST APIs",
}


# ============================================================
# TEXT HELPERS
# ============================================================

def extract_text_pdf(file):
    """Extract text from every page of an uploaded PDF."""
    reader = PdfReader(file)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    return "\n".join(pages)


def normalize_text(text):
    """Normalize text for reliable phrase/keyword matching."""
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_alias(text, alias):
    """
    Match a phrase/keyword without accidentally matching a substring
    inside an unrelated word.
    """
    alias = normalize_text(alias)
    pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def find_skills(text):
    """
    Return the skills from SKILLS that appear in a document.
    Multiple aliases for one skill count as one skill.
    """
    normalized = normalize_text(text)
    found = set()

    for skill_name, info in SKILLS.items():
        for alias in info["aliases"]:
            if contains_alias(normalized, alias):
                found.add(skill_name)
                break

    return found


def calculate_tfidf_similarity(resume_text, jd_text):
    """
    Lexical similarity component.

    We use word unigrams + bigrams. This is still only one part of
    the final score, rather than pretending raw document similarity
    is the same thing as job suitability.
    """
    resume_clean = normalize_text(resume_text)
    jd_clean = normalize_text(jd_text)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )

    vectors = vectorizer.fit_transform([resume_clean, jd_clean])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return round(float(similarity) * 100, 1)


def score_skill_group(required_skills, resume_skills):
    """
    Weighted skill coverage score from 0-100.
    """
    if not required_skills:
        return 100.0

    total_weight = sum(SKILLS[s]["weight"] for s in required_skills)
    matched_weight = sum(
        SKILLS[s]["weight"] for s in required_skills if s in resume_skills
    )

    if total_weight == 0:
        return 0.0

    return round((matched_weight / total_weight) * 100, 1)


def detect_project_relevance(resume_text):
    """
    Lightweight evidence score based on AI-project language.
    This is intentionally transparent rather than an opaque model.
    """
    text = normalize_text(resume_text)

    evidence = {
        "AI project": ["ai application", "ai project", "generative ai"],
        "RAG project": ["rag", "retrieval-augmented generation"],
        "Chatbot": ["chatbot", "chatbots"],
        "LLM": ["llm", "llms", "language model"],
        "Vector retrieval": ["pinecone", "vector database", "embeddings"],
    }

    matched = [
        label
        for label, aliases in evidence.items()
        if any(contains_alias(text, alias) for alias in aliases)
    ]

    score = min(100.0, len(matched) / len(evidence) * 100)
    return round(score, 1), matched


def detect_education_match(resume_text):
    """Check whether the resume contains an engineering/computing degree."""
    text = normalize_text(resume_text)

    education_terms = [
        "computer engineering",
        "computer science",
        "information technology",
        "engineering",
        "bachelor of engineering",
        "bachelor of technology",
        "b.tech",
        "b.e.",
    ]

    return 100.0 if any(contains_alias(text, term) for term in education_terms) else 0.0


def detect_experience_evidence(resume_text):
    """Check for internship/work/project evidence."""
    text = normalize_text(resume_text)

    terms = [
        "intern",
        "internship",
        "experience",
        "developed",
        "built",
        "implemented",
        "project",
        "hackathon",
    ]

    matches = sum(1 for term in terms if contains_alias(text, term))

    if matches >= 5:
        return 100.0
    if matches >= 3:
        return 80.0
    if matches >= 1:
        return 60.0

    return 20.0


def calculate_final_score(resume_text, jd_text):
    """
    Hybrid ATS score.

    Weighting:
      50% - core skill coverage
      25% - preferred/technical skill coverage
      10% - project relevance
       5% - education
       5% - experience evidence
       5% - TF-IDF lexical similarity

    This makes the result a job-fit score rather than a raw
    document-similarity score.
    """
    resume_skills = find_skills(resume_text)
    jd_skills = find_skills(jd_text)

    # Use JD-detected skills, but make sure the role's explicit core
    # expectations are always considered.
    core_in_jd = set(CORE_SKILLS) & jd_skills
    if not core_in_jd:
        core_in_jd = set(CORE_SKILLS)

    preferred_in_jd = set(PREFERRED_SKILLS) & jd_skills

    core_score = score_skill_group(core_in_jd, resume_skills)
    preferred_score = score_skill_group(preferred_in_jd, resume_skills)

    project_score, project_evidence = detect_project_relevance(resume_text)
    education_score = detect_education_match(resume_text)
    experience_score = detect_experience_evidence(resume_text)
    tfidf_score = calculate_tfidf_similarity(resume_text, jd_text)

    final = (
        core_score * 0.50
        + preferred_score * 0.25
        + project_score * 0.10
        + education_score * 0.05
        + experience_score * 0.05
        + tfidf_score * 0.05
    )

    final_score = max(0, min(100, round(final)))

    matched_skills = sorted(
        list(resume_skills & jd_skills),
        key=lambda x: x.lower()
    )

    missing_skills = sorted(
        list((jd_skills & (CORE_SKILLS | PREFERRED_SKILLS)) - resume_skills),
        key=lambda x: x.lower()
    )

    # Explicitly show core requirements that are absent from the JD skill
    # scan too, because they are part of this role's stated requirements.
    missing_core = sorted(
        list(set(CORE_SKILLS) - resume_skills),
        key=lambda x: x.lower()
    )

    for skill in missing_core:
        if skill not in missing_skills:
            missing_skills.append(skill)

    if final_score >= 80:
        status = "great"
        title = "Strong Match"
        message = "Your resume aligns strongly with the role's core AI engineering requirements."
    elif final_score >= 65:
        status = "moderate"
        title = "Good Match"
        message = "Your resume covers many relevant requirements, with some areas that could be strengthened."
    elif final_score >= 45:
        status = "moderate"
        title = "Partial Match"
        message = "Your resume has relevant overlap, but several important requirements are missing."
    else:
        status = "low"
        title = "Match Needs Improvement"
        message = "Your resume currently covers only a limited portion of the role's requirements."

    score_20 = round(final_score / 5)

    return {
        "score": final_score,
        "score_20": score_20,
        "status": status,
        "title": title,
        "message": message,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "breakdown": [
            {
                "name": "Core AI Skills",
                "score": round(core_score),
                "weight": "50%",
            },
            {
                "name": "Preferred Technical Skills",
                "score": round(preferred_score),
                "weight": "25%",
            },
            {
                "name": "AI Project Relevance",
                "score": round(project_score),
                "weight": "10%",
            },
            {
                "name": "Education",
                "score": round(education_score),
                "weight": "5%",
            },
            {
                "name": "Experience Evidence",
                "score": round(experience_score),
                "weight": "5%",
            },
            {
                "name": "Text Similarity",
                "score": round(tfidf_score),
                "weight": "5%",
            },
        ],
        "project_evidence": project_evidence,
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template(
            "home.html",   # <-- Changed to match the provided HTML filename
            result=None,
        )

    resume_pdf = request.files.get("resume_pdf")
    jd_pdf = request.files.get("jd_pdf")

    if not resume_pdf or not jd_pdf:
        return render_template(
            "home.html",
            result=None,
            error="Please upload both the resume PDF and job description PDF.",
        )

    try:
        resume_text = extract_text_pdf(resume_pdf)
        jd_text = extract_text_pdf(jd_pdf)
    except Exception as exc:
        return render_template(
            "home.html",
            result=None,
            error=f"Could not read one of the PDFs: {exc}",
        )

    if not resume_text.strip() or not jd_text.strip():
        return render_template(
            "home.html",
            result=None,
            error="Could not extract readable text from one or both PDFs.",
        )

    try:
        result = calculate_final_score(resume_text, jd_text)
    except Exception as exc:
        return render_template(
            "home.html",
            result=None,
            error=f"Analysis failed: {exc}",
        )

    return render_template(
        "home.html",
        result=result,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )