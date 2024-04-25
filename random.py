import tkinter as tk
from tkinter import ttk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to analyze sentiment and extract features
def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        return "Positive", sentiment_score
    elif sentiment_score < 0:
        return "Negative", sentiment_score
    else:
        return "Neutral", sentiment_score

def analyze():
    try:
        # Load Kaggle dataset
        kaggle_data = pd.read_csv("C:\\Users\\hp\\Downloads\\archive (1)\\Tweets.csv", encoding='latin1')
        
        # Select 20 random tweets
        random_tweets = kaggle_data.sample(n=min(20, len(kaggle_data)))
        
        # Prepare data for pie chart
        sentiment_counts = random_tweets['sentiment'].value_counts()
        
        # Prepare data for linear regression
        X = random_tweets['text'].apply(analyze_sentiment).apply(lambda x: x[1]).values.reshape(-1, 1)
        y = random_tweets['sentiment'].apply(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negative' else 0))
        
        # Display results including analyzed tweets
        result_text.delete("1.0", tk.END)  # Clear previous results
        
        for i, tweet in enumerate(random_tweets['text']):
            sentiment_category, sentiment_score = analyze_sentiment(tweet)
            result_text.insert(tk.END, f'Tweet {i+1}: {tweet}\n')
            result_text.insert(tk.END, f'Sentiment: {sentiment_category} (Score: {sentiment_score:.2f})\n\n')
        
        # Visualization - Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Overall Sentiment Distribution')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
        
        # Visualization - Linear regression chart
        plt.figure(figsize=(8, 6))
        sns.regplot(data=random_tweets, x=X.flatten(), y=y, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
        plt.title('Linear Regression: Sentiment vs. Sentiment Score')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Sentiment (Positive, Negative, Neutral)')
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print("An error occurred:", e)

# Tkinter app
root = tk.Tk()
root.title('Sentiment Analysis')

# Input area
input_frame = ttk.Frame(root)
input_frame.pack(padx=10, pady=10, fill='both', expand=True)

ttk.Label(input_frame, text='Click the button to analyze sentiment:').pack()

# Analyze button
analyze_button = ttk.Button(input_frame, text='Analyze Sentiment', command=analyze)
analyze_button.pack(pady=10)

# Result area
result_frame = ttk.Frame(root)
result_frame.pack(padx=10, pady=10, fill='both', expand=True)

ttk.Label(result_frame, text='Sentiment Analysis Results:').pack()
result_text = tk.Text(result_frame, height=20)
result_text.pack(fill='both', expand=True)

root.mainloop()
