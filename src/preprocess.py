import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import os
import re

# We will use scikit-learn's english stop words for simplicity and consistency
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except ImportError:
    ENGLISH_STOP_WORDS = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])

def preprocess_text(text: str) -> str:
    """
    Standard preprocessing for both documents and queries:
    1. Lowercase
    2. Normalize whitespace
    3. Remove punctuation
    4. Tokenize
    5. Remove English stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 3. Remove punctuation (we do this before whitespace normalization)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 2 & 4. Normalize whitespace and tokenize
    tokens = text.split()
    
    # 5. Remove English stopwords
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    
    return " ".join(tokens)


def main():
    raw_path = os.path.join("data", "raw", "BBCNews.csv")
    processed_path = os.path.join("data", "processed", "processed_dataset.csv")
    graphs_dir = os.path.join("outputs", "graphs")
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Rename columns based on requirements
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "document_id", "descr": "text", "tags": "tags"})
    else:
        # Fallback if structure is slightly different
        columns = list(df.columns)
        if len(columns) >= 3:
            df = df.rename(columns={columns[0]: "document_id", columns[1]: "text", columns[2]: "tags"})

    print("--- Phase 1: Inspect Dataset ---")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values
    df['text'] = df['text'].fillna("")
    df['tags'] = df['tags'].fillna("")
    
    duplicate_docs = df.duplicated(subset=['text']).sum()
    print(f"Duplicate documents (based on exact text match): {duplicate_docs}")
    
    # Compute document length
    df['doc_length'] = df['text'].apply(lambda x: len(str(x).split()))
    print(f"Average document length (words): {df['doc_length'].mean():.2f}")
    print(f"Min document length: {df['doc_length'].min()}")
    print(f"Max document length: {df['doc_length'].max()}")
    
    # Tag/category distribution
    # Tags might be comma-separated, or single words. Let's see.
    all_tags = df['tags'].apply(lambda x: [tag.strip() for tag in str(x).split(',') if tag.strip()])
    tag_counts = pd.Series([tag for tag_list in all_tags for tag in tag_list]).value_counts()
    
    print(f"Unique tags: {len(tag_counts)}")
    print("Top 10 tags:")
    print(tag_counts.head(10))
    
    # Save graph for tag distribution (top 15)
    plt.figure(figsize=(10, 6))
    tag_counts.head(15).plot(kind='bar', color='skyblue')
    plt.title('Top 15 Categories/Tags in BBC News Dataset')
    plt.xlabel('Category / Tag')
    plt.ylabel('Number of Documents')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'tag_distribution.png'))
    print(f"Saved tag distribution plot to {os.path.join(graphs_dir, 'tag_distribution.png')}")
    
    print("\n--- Phase 2: Preprocessing ---")
    print("Applying preprocessing to text (this may take a few seconds)...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Drop intermediate columns if we want, but keeping doc_length is fine.
    df = df[['document_id', 'text', 'tags', 'clean_text']]
    
    print(f"Saving processed dataset to {processed_path}...")
    df.to_csv(processed_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
