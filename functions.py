import jieba

import nltk
nltk.download('punkt')
ntl.download('punkt_tab')
from nltk.tokenize import word_tokenize

from scipy.stats import chi2_contingency

import seaborn as sns

import pandas as pd

from tqdm import tqdm

import datetime as dt 

import matplotlib.pyplot as plt 

def check_negation(text: str, language: str, sentiment_dict: dict) -> list:
    """
    Check for negation words in the context and return their positions in the text.
    
    Parameters:
        text (str): The text to analyze
        language (str): The language of the text
        sentiment_dict (dict): The sentiment dictionary for the language containing negation words

    Returns:
        list: A list of positions of negation words in the text

    """

    negations = sentiment_dict[language]['negations']
    words = text.lower().split()
    negation_positions = [i for i, word in enumerate(words) if word in negations]
    return negation_positions


def analyze_sentiment(text: str, language: str, sentiment_dict: dict) -> float:
    """
    Analyze the sentiment of a text in a given language using a sentiment dictionary.
    Support of English, Russian and Simplified Chinese languages.
    
    Parameters:
        text (str): The text to analyze
        language (str): The language of the text
        sentiment_dict (dict): The sentiment dictionary for the language
            
    Returns:
        float: The sentiment score of the text (normalized between -1 and 1)
                
    """
    
    score = 0
    abs_score = 0
    
    # Get positive and negative dictionaries for the language
    pos_dict = sentiment_dict[language]['positive']
    neg_dict = sentiment_dict[language]['negative']
    
    # Find negation positions
    negation_positions = check_negation(text, language, sentiment_dict)
    
    # Tokenize the text
    text = text.lower()

    if language == 'schinese':
        words = jieba.lcut(text)
    elif language == 'russian':
        words = word_tokenize(text, language='russian')
    else:
        words = word_tokenize(text)
    
    # Analyze each word considering context and negation
    for i, word in enumerate(words):
        sentiment_value = 0
        
        # Check if word is in sentiment dictionaries
        if word in pos_dict:
            sentiment_value = pos_dict[word]
        elif word in neg_dict:
            sentiment_value = neg_dict[word]
            
        # Check if word is negated
        for neg_pos in negation_positions:
            if 0 <= i - neg_pos <= 3:  # Check if word is within 3 words after negation
                sentiment_value *= -1  # Reverse the sentiment
                
        score += sentiment_value
        abs_score += abs(sentiment_value)
    
    # Normalize score between -1 and 1
    normalized_score = score / abs_score if abs_score > 0 else 0
    
    return normalized_score

def parse_time(time_in_secs):
    """ Convert Unix timestamp to datetime """
    return pd.to_datetime(time_in_secs, unit = 's')


def plot_reviews_by_interval(df, intervals):
    """ Plot the number of reviews per time interval """
    # Convert the timestamp to datetime and extract the time
    tqdm.pandas(desc="Conversione dei timestamp")
    df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], unit='s')
    df['time'] = df['timestamp_created'].dt.time  # Estraiamo solo il tempo (HH:MM:SS)

    # Create a dictionary to store the count of reviews for each interval
    interval_counts = {interval['name']: 0 for interval in intervals}

    # Use tqdm to show a progress bar while iterating over the rows
    for index, row in tqdm(df.iterrows(), desc="Conteggio recensioni per intervallo", total=df.shape[0]):
        for interval in intervals:
            if interval['start'] <= row['time'] <= interval['end']:
                interval_counts[interval['name']] += 1
                break

    # Convert the dictionary to a DataFrame
    interval_df = pd.DataFrame(list(interval_counts.items()), columns=['Interval', 'Review Count'])

    # Plot the number of reviews per interval
    plt.figure(figsize=(10, 6))
    plt.bar(interval_df['Interval'], interval_df['Review Count'], color='skyblue')
    plt.title('Number of Reviews per Time Interval')
    plt.xlabel('Time Interval')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def analyze_sentiment_correlation(df, sentiment_col='sentiment', recommended_col='recommended'):
    """
    Analyze correlation between sentiment labels and recommendation status.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing sentiment and recommendation columns
    sentiment_col (str): Name of the sentiment column
    recommended_col (str): Name of the recommendation column
    
    Returns:
    dict: Dictionary containing correlation statistics and contingency table
    """
    # Create contingency table
    contingency_table = pd.crosstab(df[sentiment_col], df[recommended_col])
    
    # Perform chi-square test
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    # Calculate percentages within each sentiment category
    percentage_table = pd.crosstab(df[sentiment_col], df[recommended_col], normalize='index') * 100
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(percentage_table, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Sentiment vs. Recommendation Distribution (%)')
    
    results = {
        'contingency_table': contingency_table,
        'percentage_table': percentage_table,
        'chi_square': chi2,
        'p_value': p_value,
        'cramers_v': cramer_v
    }
    
    return results