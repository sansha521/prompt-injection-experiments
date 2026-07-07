from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from load_data import load_data
from evaluate import evaluate
import time


MODEL_NAME = "multinomial_naive_bayes"
feature_type = "bag_of_words"
vectorizer = "CountVectorizer"
num_samples = 100



def load_classifier(X_train, y_train):
    # Convert text to bag-of-words features
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2))
    X_train_counts = vectorizer.fit_transform(X_train)

    # Train classifier
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_counts, y_train)

    return vectorizer, model    

def predict(vectorizer, model, X):
    return model.predict(vectorizer.transform(X))


print("Preparing data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
print("Data ready.")

print("Loading classifier...")
vectorizer, classifier = load_classifier(X_train, y_train)
print("Classifier ready.")

print("Making predictions...")
start = time.perf_counter()

predictions = predict(vectorizer, classifier, X_train[:])

end = time.perf_counter()

total_latency = end - start
print("Predictions ready.")

truth = y_test[:]

print("Evaluating...")
evaluate(truth, predictions, MODEL_NAME, feature_type, vectorizer, total_latency, len(truth))
print("Metrics ready.")