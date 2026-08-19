from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def clean_function(text):
    text = text.lower()
    text = nlp(text)

    text = [t for t in text]
    text = [t for t in text if not t.is_punct]
    text = [t for t in text if not t.is_stop]
    text = [t.lemma_ for t in text]
    text = [str(t) for t in text]

    text = " ".join(text)

    return text


app = Flask(__name__)


def extract_text_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        text = text + page.extract_text()

    return text


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get uploaded files
        resume_pdf = request.files.get("resume_pdf")
        jd_pdf = request.files.get("jd_pdf")

        # Extract text from PDFs
        resume_text = extract_text_pdf(resume_pdf)
        jd_text = extract_text_pdf(jd_pdf)

        # Preprocess text
        clean_resume = clean_function(resume_text)
        clean_jd = clean_function(jd_text)

        # Convert documents into TF-IDF vectors
        tv = TfidfVectorizer()

        vectors = tv.fit_transform([
            clean_resume,
            clean_jd
        ])

        # Calculate cosine similarity
        similarity = cosine_similarity(
            vectors[0],
            vectors[1]
        )

        similarity = similarity[0][0]

        # Convert similarity to percentage
        percentage_score = round(similarity * 100)

        # Convert percentage score to score out of 20
        score_out_of_20 = round(similarity * 20)

        # Make sure score stays within 0-20
        score_out_of_20 = max(0, min(20, score_out_of_20))

        # Classify the result
        if score_out_of_20 >= 14:

            status = "great"

            title = "Great Match!"

            message = (
                "Your resume shows strong textual alignment "
                "with this job."
            )

        elif score_out_of_20 >= 10:

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

        return render_template(
            "home.html",
            status=status,
            title=title,
            message=message,
            score=score_out_of_20,
            percentage=percentage_score
        )

    else:

        return render_template(
            "home.html",
            status="",
            title="",
            message="",
            score=None,
            percentage=None
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )






# # from flask import *
# # from PyPDF2 import PdfReader

# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity


# # import spacy 
# # nlp = spacy.load("en_core_web_lg")

# # def clean_function(text):
# #     text = text.lower()
# #     text = nlp.text()
# #     text = [t for t in text]
# #     text = [t for t in text if not t.is_punct]
# #     text = [t for t in text if not t.is_stop]
# #     text = [t.lemma_ for t in text]
# #     text = [str(t) for t in text]  
# #     text = " ".join(text)
# #     return text

# # app = Flask(__name__)

# # def extract_text_pdf(pdf):
# #     reader = PdfReader(pdf)
# #     text = ""
# #     for page in reader.pages:
# #         text = text + page.extract_text()
# #     return text

# # @app.route("/", methods=["GET", "POST"])
# # def home():
# #     if request.method == "POST":
# #         resume_pdf = request.files.get("resume_pdf")
# #         jd_pdf = request.files.get("jd_pdf")

# #         resume_text = extract_text_pdf(resume_pdf)
# #         jd_text = extract_text_pdf(jd_pdf)

# #         print(resume_text)
# #         print("________________________________________________")
# #         print(jd_text)

# #         clean_resume = clean_function(resume_text)
# #         clean_jd = clean_function(jd_text)

# #         tv = TfidfVectorizer()
# #         vectors = tv.fit_transform([clean_resume, clean_jd])
# #         score = cosine_similarity(vectors[0], vectors[1])
# #         score = score[0][0]
# #         score = round(score, 2) * 100
# #         msg = "YOur score is: " + str(score) 
# #         return render_template("home.html", msg=msg)

# #     else:
# #         return render_template("home.html")    

# # if __name__ == "__main__":
# #     app.run(debug=True,use_reloader=True)

# #==============================================================
# from flask import *
# from PyPDF2 import PdfReader
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import spacy
# # nlp = spacy.load("en_core_web_lg")
# nlp = spacy.load("en_core_web_sm")


# def clean_function(text):
#     text = text.lower()
#     text = nlp(text)
#     text = [t for t in text]
#     text = [t for t in text if not t.is_punct]
#     text = [t for t in text if not t.is_stop]
#     text = [t.lemma_  for t in text]
#     text = [str(t) for t in text]
#     text = " ".join(text)
#     return text
    

# app = Flask(__name__)

# def extract_text_pdf(pdf):
#     reader = PdfReader(pdf)
#     text=""
#     for page in reader.pages:
#         text = text+page.extract_text()
#     return text

# @app.route("/",methods=["GET","POST"])
# def home():
#     if request.method == "POST":
#         resume_pdf = request.files.get("resume_pdf")
#         jd_pdf = request.files.get("jd_pdf")
        
#         resume_text = extract_text_pdf(resume_pdf)
#         jd_text = extract_text_pdf(jd_pdf)
        
#         clean_resume = clean_function(resume_text)
#         clean_jd = clean_function(jd_text)
#         tv = TfidfVectorizer()
#         vectors = tv.fit_transform([clean_resume,clean_jd])
#         score = cosine_similarity(vectors[0],vectors[1])
#         score = score[0][0]
#         score = round(score,2)*100
#         msg = "Your score is "+str(score)

#         if score >= 70:
#             status = "great"
#             msg = f"Great Match! Your resume is a strong match for this job. Match Score: {score:.0f}%"
#         elif score >= 50:
#             status = "moderate"
#             msg = f"Moderate Match. Your resume has a partial match with this job. Match Score: {score:.0f}%"
#         else:
#             status = "low"
#             msg = f"Not a Strong Match. Your resume has limited similarity with this job. Match Score: {score:.0f}%"
            
#         return render_template("home.html",msg=msg)
        
#     else:
#         return render_template("home.html")
    
# # if __name__ == "__main__":
# #     app.run(debug=True,use_reloader=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

