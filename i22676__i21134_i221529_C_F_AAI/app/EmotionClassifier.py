
# Handles emotion classification from text
# Contains the EmotionClassifier class with training and prediction methods
# Uses sklearn for text classification with a TF-IDF vectorizer and LogisticRegression


import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EmotionClassifier:
    """
    A classifier that predicts emotions from text based on training data
    """
    def __init__(self):
        # Define the emotion labels
        self.emotions = ['sadness', 'joy', 'fear', 'anger', 'love', 'neutral']
        
        # Create a pipeline with TF-IDF vectorizer and LogisticRegression classifier
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
            ('clf', LogisticRegression(max_iter=1000, C=1.0))
        ])
        
        self.is_trained = False
    
    # training based on emotions
    #  taking the data file and work based on it
    def load_data(self, filepath):
        
        # Load data from text file in the format: "text;emotion"
    
        texts = []
        emotions = []
        
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # Split by the last semicolon (some texts might contain semicolons)
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    # The last part is the emotion
                    emotion = parts[-1]
                    # Join everything else as the text
                    text = ';'.join(parts[:-1])
                    
                    texts.append(text)
                    emotions.append(emotion)
                    
        return texts, emotions
    
    def train(self, train_file, val_file=None):
        """
        Train the emotion classifier using training data
        """
        print("Loading training data...")
        X_train, y_train = self.load_data(train_file)
        
        print(f"Training on {len(X_train)} examples...")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Validate if validation file is provided
        if val_file:
            X_val, y_val = self.load_data(val_file)
            accuracy = self.pipeline.score(X_val, y_val)
            print(f"Validation accuracy: {accuracy:.4f}")

        return self
    
    def predict(self, text):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
            ('clf', LogisticRegression(max_iter=1000, C=1.0))
        ])
        self.is_trained = False

    def predict(self, text):
        """Returns: (emotion_str, confidence_dict)"""
        if not self.is_trained:
            return "neutral", {"neutral": 1.0} 
            
        emotion = self.pipeline.predict([text])[0]
        probs = self.pipeline.predict_proba([text])[0]
        emotion_probs = dict(zip(self.pipeline.classes_, probs))
        
        if max(emotion_probs.values()) < 0.5:
            return "neutral", emotion_probs
        return emotion, emotion_probs