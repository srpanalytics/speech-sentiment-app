import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to analyze sentiment
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    for idx, para in enumerate(paragraphs, 1):
        score = analyzer.polarity_scores(para)
        results.append({
            'Paragraph': idx,
            'Text': para,
            'Compound': score['compound'],
            'Positive': score['pos'],
            'Neutral': score['neu'],
            'Negative': score['neg']
        })
    return results

# Function to generate PDF report
def generate_pdf(sentiment_results):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Sentiment Analysis Report")
    y -= 30

    c.setFont("Helvetica", 10)
    for r in sentiment_results:
        lines = [
            f"Paragraph {r['Paragraph']} (Compound: {r['Compound']:.2f}):",
            r['Text'],
            "-"*80
        ]
        for line in lines:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(50, y, line[:100])  # truncate long lines
            y -= 15

    c.save()
    return temp_pdf.name

# Streamlit UI
st.title("Speech Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload your speech (PDF format)", type=["pdf"])

if uploaded_file:
    if st.button("▶️ Run Sentiment Analysis"):
        with st.spinner("Analyzing..."):
            text = extract_text_from_pdf(uploaded_file)
            results = analyze_sentiment(text)
            pdf_path = generate_pdf(results)
        st.success("Analysis Complete ✅")

        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Sentiment Report (PDF)", f, file_name="sentiment_report.pdf")

        os.remove(pdf_path)  # optional cleanup
