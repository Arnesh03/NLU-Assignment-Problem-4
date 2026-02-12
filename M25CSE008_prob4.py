import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Problem 4: Sports vs Politics Classifier with Comprehensive Analysis
# Roll Number: M25CSE008

def print_dataset_stats(X, y):
    # just print some basics about the data
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total documents: {len(X)}")
    print(f"Sports documents: {y.count('Sports')}")
    print(f"Politics documents: {y.count('Politics')}")
    
    # Calculate average document length
    lengths = [len(doc.split()) for doc in X]
    print(f"\nAverage document length: {np.mean(lengths):.1f} words")
    print(f"Min length: {min(lengths)} words")
    print(f"Max length: {max(lengths)} words")
    print(f"Median length: {np.median(lengths):.1f} words")

def compare_vectorizers(X_train, X_test, y_train, y_test):
    # check BoW vs TF-IDF to see which one is better
    print("\n" + "="*50)
    print("FEATURE REPRESENTATION COMPARISON")
    print("="*50)
    
    # Test with Bag of Words (CountVectorizer)
    bow_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english', max_features=5000)),
        ('clf', MultinomialNB())
    ])
    bow_pipeline.fit(X_train, y_train)
    bow_pred = bow_pipeline.predict(X_test)
    bow_acc = accuracy_score(y_test, bow_pred)
    
    # Test with TF-IDF
    tfidf_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', MultinomialNB())
    ])
    tfidf_pipeline.fit(X_train, y_train)
    tfidf_pred = tfidf_pipeline.predict(X_test)
    tfidf_acc = accuracy_score(y_test, tfidf_pred)
    
    print(f"Bag of Words (CountVectorizer) Accuracy: {bow_acc:.4f}")
    print(f"TF-IDF Vectorizer Accuracy: {tfidf_acc:.4f}")
    print(f"\nTF-IDF performs {'better' if tfidf_acc > bow_acc else 'worse'} by {abs(tfidf_acc - bow_acc):.4f}")

def main():
    print("Loading the 20 Newsgroups dataset...")
    
    # categories we need for the assignment
    categories = [
        'rec.sport.baseball', 'rec.sport.hockey',
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'
    ]
    
    # Download the data
    try:
        data = fetch_20newsgroups(subset='all', categories=categories, 
                                  remove=('headers', 'footers', 'quotes'))
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Documents loaded: {len(data.data)}")
    
    # Prepare binary labels
    X_text = []
    y_labels = []
    
    sport_idx = [i for i, name in enumerate(data.target_names) if 'sport' in name]
    politics_idx = [i for i, name in enumerate(data.target_names) if 'politics' in name]
            
    for doc, label in zip(data.data, data.target):
        if label in sport_idx:
            X_text.append(doc)
            y_labels.append('Sports')
        elif label in politics_idx:
            X_text.append(doc)
            y_labels.append('Politics')
    
    # Print dataset statistics
    print_dataset_stats(X_text, y_labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_labels, test_size=0.2, random_state=42
    )
    
    # Compare BoW vs TF-IDF
    compare_vectorizers(X_train, X_test, y_train, y_test)
    
    # Define models to compare
    models = [
        ('Naive Bayes', MultinomialNB()),
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('Linear SVM', SGDClassifier(max_iter=1000, tol=1e-3))
    ]
    
    print("\n" + "="*50)
    print("MODEL COMPARISON (Using TF-IDF)")
    print("="*50)
    
    results = {}
    
    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print('='*50)
        
        # Create pipeline with TF-IDF
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
            ('clf', model),
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        results[name] = acc
        
        print(f"\nAccuracy: {acc:.4f}")
        
        # Confusion Matrix stuff
        cm = confusion_matrix(y_test, predictions, labels=['Politics', 'Sports'])
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Politics  Sports")
        print(f"Actual Politics    {cm[0][0]:4d}    {cm[0][1]:4d}")
        print(f"       Sports      {cm[1][0]:4d}    {cm[1][1]:4d}")
        
        # Classification Report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, predictions))
    
    # Final Summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:25s}: {acc:.4f}")
    
    best_model = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best_model} ({results[best_model]:.4f})")
    
    # Vocabulary size info
    print("\n" + "="*50)
    print("FEATURE SPACE INFORMATION")
    print("="*50)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    vectorizer.fit(X_train)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Feature limit: 10000")
    print(f"Stop words removed: Yes (English)")

if __name__ == "__main__":
    main()
