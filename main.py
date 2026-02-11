import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('punkt')

# --- CONFIGURATION ---
DATA_PATH = "all_tickets_processed_improved_v3.csv"
MAX_FEATURES = 5000  # Max words for TF-IDF

# --- 1. DATA LOADING & TEXT CLEANING ---
def clean_text(text):
    """
    Client Requirement: Lowercasing, Stopword Removal, Punctuation Handling
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Punctuation & Special Chars (Keep only a-z and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Stopword Removal
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    
    return " ".join(filtered_words)

def assign_synthetic_priority(text):
    """
    Client Requirement: Priority Prediction (High / Medium / Low).
    Since the dataset lacks a 'Priority' column, we generate one based on keywords.
    """
    text = text.lower()
    if any(x in text for x in ['urgent', 'critical', 'immediately', 'outage', 'failure', 'deadline']):
        return 'High'
    elif any(x in text for x in ['error', 'issue', 'warning', 'slow', 'problem', 'unable']):
        return 'Medium'
    else:
        return 'Low'

print("⏳ Loading and Cleaning Data...")
df = pd.read_csv(DATA_PATH)

# Rename columns to standard names if needed
if 'Document' in df.columns:
    df.rename(columns={'Document': 'text', 'Topic_group': 'category'}, inplace=True)

# Apply Cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Generate Priority Labels (Rule-based)
df['priority'] = df['text'].apply(assign_synthetic_priority)

print(f"✅ Data Prepared: {len(df)} records")
print(f"   Categories: {df['category'].nunique()}")
print(f"   Priorities: {df['priority'].value_counts().to_dict()}")

# --- 2. FEATURE EXTRACTION (TF-IDF) ---
print("\n⚙️  Extracting Features (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
X = tfidf.fit_transform(df['clean_text']).toarray()

# --- 3. MODEL TRAINING & EVALUATION ---
def train_and_evaluate(X, y, target_name):
    print(f"\n🚀 Training Model for: {target_name.upper()}...")
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection: Random Forest is robust for text classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2%}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # BONUS: Confusion Matrix (SAVING LOGIC ADDED HERE)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix: {target_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    # SAVE THE PLOT
    filename = f"confusion_matrix_{target_name}.png"
    plt.savefig(filename)
    print(f"💾 Confusion Matrix saved to: {filename}")
    plt.close() # Close the plot to free memory
    
    return model

# Train Category Classifier
category_model = train_and_evaluate(X, df['category'], "Category")

# Train Priority Classifier
priority_model = train_and_evaluate(X, df['priority'], "Priority")

# --- 4. PREDICTION FUNCTION ---
def predict_new_ticket(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned]).toarray()
    
    cat_pred = category_model.predict(vectorized)[0]
    pri_pred = priority_model.predict(vectorized)[0]
    
    return cat_pred, pri_pred

# Example usage
print("\n🤖 Test Prediction:")
test_ticket = "URGENT: Server outage in data center, cannot access files!"
cat, pri = predict_new_ticket(test_ticket)
print(f"Ticket: '{test_ticket}'")
print(f"Predicted Category: {cat}")
print(f"Predicted Priority: {pri}")