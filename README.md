# ATS Resume Matcher

A Flask-based NLP web application that compares a resume with a job description and generates a compatibility score using text preprocessing, TF-IDF vectorization, and cosine similarity.

# ATS Resume Matcher

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/PyPDF2-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white" />
  <img src="https://img.shields.io/badge/Render-000000?style=flat-square&logo=render&logoColor=white" />
</p>

<p align="center">
  <strong>📄 PDF Processing</strong> •
  <strong>🧠 NLP</strong> •
  <strong>📊 TF-IDF</strong> •
  <strong>📐 Cosine Similarity</strong> •
  <strong>🌐 Flask Web App</strong>
</p>

A Flask-based NLP web application that compares a resume with a job description and generates a compatibility score using text preprocessing, TF-IDF vectorization, and cosine similarity.

## 🚀 Overview

ATS Resume Matcher helps users evaluate how closely their resume matches a given job description.

The application accepts two PDF files:

* Resume
* Job Description

It extracts and preprocesses the text from both documents, converts the processed text into TF-IDF vectors, calculates cosine similarity, and displays the resulting match score.

## Architecture

<img src="images/ATS RESUME Matcher.png" />

## ✨ Features

* Upload resume as a PDF
* Upload job description as a PDF
* Extract text from PDF documents
* NLP-based text preprocessing using spaCy
* Lowercase normalization
* Stop-word removal
* Punctuation removal
* Lemmatization
* TF-IDF vectorization
* Cosine similarity calculation
* Resume-JD compatibility score
* Responsive web interface
* Flask-based backend
* Deployable as a Python web application

## 🧠 How It Works

The application follows the pipeline below:

```text
Resume PDF ───────┐
                  │
                  ▼
             PDF Text Extraction
                  │
                  ▼
             NLP Preprocessing
        ┌─────────┼─────────┐
        │         │         │
   Stop Words  Punctuation  Lemmatization
        └─────────┼─────────┘
                  ▼
             TF-IDF Vectors
                  │
                  ▼
          Cosine Similarity
                  │
                  ▼
             Match Score
```

### 1. PDF Text Extraction

The application uses `PyPDF2` to extract text from the uploaded resume and job-description PDFs.

### 2. Text Preprocessing

The extracted text is processed using spaCy.

The preprocessing pipeline includes:

* Converting text to lowercase
* Removing punctuation
* Removing stop words
* Lemmatizing words

### 3. TF-IDF Vectorization

The cleaned resume and job-description text are converted into numerical representations using `TfidfVectorizer` from scikit-learn.

TF-IDF helps represent the importance of terms within the two documents.

### 4. Cosine Similarity

The application calculates cosine similarity between the resume and job-description vectors.

The similarity value is converted into a percentage-style compatibility score and displayed to the user.

## 🛠️ Tech Stack

### Backend

* Python
* Flask

### NLP / Machine Learning

* spaCy
* scikit-learn
* TF-IDF
* Cosine Similarity

### PDF Processing

* PyPDF2

### Frontend

* HTML5
* CSS3
* Font Awesome
* Google Fonts

### Deployment

* Render
* Gunicorn

## 📁 Project Structure

```text
ATS-Resume-Matcher/
│
├── templates/
│   └── home.html
│
├── app.py
├── requirements.txt
├── Procfile
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/anaydeshpande1749/ATS-Resume-Matcher.git
cd ATS-Resume-Matcher
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the spaCy language model

The application uses the spaCy English small language model.

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the application

```bash
python app.py
```

The application will run locally on:

```text
http://localhost:5000
```

## 🌐 Deployment

The application is deployed as a Flask web service on Render using Gunicorn.

A typical Render deployment uses:

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
gunicorn app:app
```

Render supports deploying Flask applications as Python web services and recommends Gunicorn for production serving.

## 📊 Example Workflow

1. Upload your resume PDF.
2. Upload the target job description PDF.
3. Click **Check Match**.
4. The application extracts text from both documents.
5. The text is cleaned and lemmatized.
6. TF-IDF vectors are generated.
7. Cosine similarity is calculated.
8. The resulting compatibility score is displayed.

## 🔮 Future Improvements

Possible improvements include:

* Keyword-level matching
* Missing-skill identification
* Section-wise resume analysis
* Skill extraction
* Job-description keyword extraction
* Weighted skill matching
* Resume recommendations
* Highlighting matched and missing terms
* More advanced semantic similarity models
* Improved handling of different PDF formats

## ⚠️ Limitations

The current implementation primarily measures textual similarity between the resume and job description.

A higher similarity score does not necessarily mean that a resume will pass a real Applicant Tracking System or guarantee job suitability.

The quality of extracted text can also depend on the structure and formatting of the uploaded PDFs.

## 📄 License

This project is licensed under the MIT License.
