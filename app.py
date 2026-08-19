from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


# --------------------------------------------------
# Load spaCy English model
# --------------------------------------------------

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def clean_function(text):

    text = text.lower()

    doc = nlp(text)

    tokens = []

    for token in doc:

        # Remove punctuation
        if token.is_punct:
            continue

        # Remove stop words
        if token.is_stop:
            continue

        # Add lemmatized word
        tokens.append(token.lemma_)

    return " ".join(tokens)


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Extract text from PDF
# --------------------------------------------------

def extract_text_pdf(pdf):

    reader = PdfReader(pdf)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    # ----------------------------------------------
    # First visit
    # ----------------------------------------------

    if request.method == "GET":

        return render_template(
            "home.html",
            score=None,
            percentage=None,
            status="",
            title="",
            message=""
        )


    # ----------------------------------------------
    # Get uploaded files
    # ----------------------------------------------

    resume_pdf = request.files.get("resume_pdf")
    jd_pdf = request.files.get("jd_pdf")


    # ----------------------------------------------
    # Validate files
    # ----------------------------------------------

    if not resume_pdf or not jd_pdf:

        return render_template(
            "home.html",
            score=None,
            percentage=None,
            status="",
            title="",
            message="Please upload both PDF files."
        )


    # ----------------------------------------------
    # Extract PDF text
    # ----------------------------------------------

    resume_text = extract_text_pdf(resume_pdf)

    jd_text = extract_text_pdf(jd_pdf)


    # ----------------------------------------------
    # Validate extracted text
    # ----------------------------------------------

    if not resume_text.strip() or not jd_text.strip():

        return render_template(
            "home.html",
            score=None,
            percentage=None,
            status="",
            title="",
            message="Could not extract enough text from one or both PDFs."
        )


    # ----------------------------------------------
    # NLP preprocessing
    # ----------------------------------------------

    clean_resume = clean_function(resume_text)

    clean_jd = clean_function(jd_text)


    # ----------------------------------------------
    # TF-IDF Vectorization
    # ----------------------------------------------

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [
            clean_resume,
            clean_jd
        ]
    )


    # ----------------------------------------------
    # Cosine Similarity
    # ----------------------------------------------

    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0][0]


    # ----------------------------------------------
    # Convert similarity to percentage
    # ----------------------------------------------

    percentage = round(similarity * 100)


    # ----------------------------------------------
    # Convert score to /20
    # ----------------------------------------------

    score = round(similarity * 20)


    # Make sure score stays between 0 and 20

    score = max(0, min(20, score))


    # ----------------------------------------------
    # Classify result
    # ----------------------------------------------

    if score >= 14:

        status = "great"

        title = "Great Match!"

        message = (
            "Your resume shows strong textual alignment "
            "with this job."
        )


    elif score >= 10:

        status = "moderate"

        title = "Moderate Match"

        message = (
            "Your resume shows partial textual alignment "
            "with this job."
        )


    else:

        status = "low"

        title = "Match Needs Improvement"

        message = (
            "Your resume shows limited textual alignment "
            "with this job."
        )


    # ----------------------------------------------
    # Send result to HTML
    # ----------------------------------------------

    return render_template(
        "home.html",

        score=score,

        percentage=percentage,

        status=status,

        title=title,

        message=message
    )


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )