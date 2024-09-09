from transformers import pipeline
import pandas as pd

file_path = 'Path to file/file.csv'# Set the path to csv file to process it
sentiment_pipeline = pipeline("sentiment-analysis")
df = pd.read_csv(file_path)
texts = df['review'].tolist()
results = sentiment_pipeline(texts)
for i, result in enumerate(results):
    print(f"Review {i+1}: {result['label']} (Confidence: {result['score']:.2f})")