import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os
import matplotlib.pyplot as plt

# 1. Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# 2. Sentiment Analysis
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

# 3. Plot and save graph
def plot_sentiment(results):
    x = [r['Paragraph'] for r in results]
    y = [r['Compound'] for r in results]

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='o', color='blue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Sentiment Trend (Compound Score by Paragraph)')
    plt.xlabel('Paragraph')
    plt.ylabel('Compound Score')
    plt.tight_layout()

    graph_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# 4. Generate PDF with graph + text
def generate_pdf(sentiment_results, graph_path):
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Sentiment Analysis Report")
    y -= 30

    # Graph
    c.drawImage(ImageReader(graph_path), 50, y - 250, width=500, height=250)
    y -= 270

    # Sentiment text summary
    c.setFont("Helvetica", 9)
    for r in sentiment_results:
        lines = [
            f"Paragraph {r['Paragraph']} (Compound: {r['Compound']:.2f}):",
            r['Text'],
            "-" * 90
        ]
        for line in lines:
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 9)
            c.drawString(50, y, line[:100])
            y -= 13

    c.save()
    return pdf_path

# 5. Streamlit App UI
st.title("📊 PDF Speech Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload your speech (PDF format)", type=["pdf"])

if uploaded_file:
    if st.button("▶️ Run Sentiment Analysis"):
        with st.spinner("Analyzing..."):
            text = extract_text_from_pdf(uploaded_file)
            results = analyze_sentiment(text)
            graph_img = plot_sentiment(results)
            pdf_report = generate_pdf(results, graph_img)

        st.success("✅ Analysis Complete")

        with open(pdf_report, "rb") as f:
            st.download_button("📥 Download Sentiment Report (with Graph)", f, file_name="sentiment_report.pdf")

        # Optional: cleanup
        os.remove(graph_img)
        os.remove(pdf_report)
