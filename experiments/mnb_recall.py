import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score

from load_data import load_data
from evaluate import evaluate
import time


MODEL_NAME = "multinomial_naive_bayes_tuned"
feature_type = "bag_of_words"


def load_classifier(X_train, y_train):

    # Pipeline: vectorizer + model
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("model", MultinomialNB())
    ])

    # Hyperparameter grid
    param_grid = {
        "vectorizer__stop_words": [None, "english"],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "model__alpha": [0.1, 0.5, 1.0, 2.0]
    }

    fbeta = make_scorer(
        fbeta_score,
        beta=2,         # larger beta -> more recall 3, 5
        pos_label=1
    )

    # Grid search
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring=fbeta,
        n_jobs=-1,
        refit=True
    )

    # Train
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)

    # Return best model
    return grid.best_estimator_


def predict(model, X, y):
    probs = model.predict_proba(X)
    preds = model.predict(X)

    y_true = y

    false_mask = preds != y_true

    false_samples = X[false_mask]
    false_probs = probs[false_mask]
    false_preds = preds[false_mask]
    false_true = y_true[false_mask]

    false_confidence = false_probs.max(axis=1)

    stats = {
        "probs": false_probs,
        "preds": false_preds,
        "true": false_true,
        "confidence": false_confidence,
    }
    # print(f"Misclassified samples: {len(false_true)}")
    # print(stats["confidence"][:10])

    for sample, pred, true, conf in zip(
        false_samples,
        false_preds,
        false_true,
        false_confidence,
    ):
        print(f"Text      : {sample}")
        print(f"Predicted : {pred}")
        print(f"True      : {true}")
        print(f"Confidence: {conf:.3f}")
        print("-" * 80)

    return preds, stats


print("Preparing data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
X_train = pd.concat([X_train, X_val], ignore_index=True)
y_train = pd.concat([y_train, y_val], ignore_index=True)
print("Data ready.")

print("Loading classifier...")
model = load_classifier(X_train, y_train)
print("Classifier ready.")

print("Making predictions...")
start = time.perf_counter()

predictions, stats = predict(model, X_test, y_test)

end = time.perf_counter()

total_latency = end - start
print("Predictions ready.")

truth = y_test

print("Evaluating...")
evaluate(
    truth,
    predictions,
    MODEL_NAME,
    feature_type,
    model.named_steps["vectorizer"],
    total_latency,
    len(truth)
)

print("Metrics ready.")

# def predict_text(model, text):
#     pred = model.predict([text])[0]
#     probs = model.predict_proba([text])[0]

#     print(f"Text: {text}")
#     print(f"Prediction: {pred}")
#     print(f"Confidence: {probs.max():.3f}")

