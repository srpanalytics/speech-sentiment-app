import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use('Agg')  # Avoids tkinter-related errors
import matplotlib.pyplot as plt

# 1. Load text from a .txt file
def load_text_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# 2. Analyze sentiment paragraph by paragraph
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    results = []

    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    for idx, para in enumerate(paragraphs, 1):
        score = analyzer.polarity_scores(para)
        results.append({
            'Paragraph': idx,
            'Text': para,
            'Positive': score['pos'],
            'Neutral': score['neu'],
            'Negative': score['neg'],
            'Compound': score['compound']
        })
    return results

# 3. Plot compound sentiment scores and save as image
def plot_sentiment(results):
    x = [r['Paragraph'] for r in results]
    y = [r['Compound'] for r in results]

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='blue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Sentiment Trend Across Speech')
    plt.xlabel('Paragraph')
    plt.ylabel('Compound Sentiment Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sentiment_plot.png")
    print("📊 Sentiment plot saved as 'sentiment_plot.png'")

# 4. Print sentiment summary to console
def print_summary(results):
    print("\n📋 Sentiment Analysis Summary:")
    for r in results:
        print(f"\nParagraph {r['Paragraph']} (Compound: {r['Compound']}):")
        print(f"Text: {r['Text']}")

# 5. Main execution
if __name__ == "__main__":
    file_path = "speech_report.txt"  # ✅ Change this to your actual file
    try:
        text_data = load_text_file(file_path)
        sentiment_results = analyze_sentiment(text_data)
        print_summary(sentiment_results)
        plot_sentiment(sentiment_results)
    except Exception as e:
        print(f"❌ Error: {e}")
