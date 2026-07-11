print("Starting...")

print("Importing pandas...")
import pandas as pd
print("Importing numpy...")
import numpy as np

print("Importing scikit-learn libraries...")
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score

print("Importing load data...")
from load_data import load_data
print("Importing evaluate...")
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
        "model__alpha": [0.1, 0.5, 1.0, 2.0],
        "vectorizer__min_df": [1, 2, 5]
    }

    fbeta = make_scorer(
        fbeta_score,
        beta=3,         # larger beta -> more recall 3, 5
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
    print("Best CV F-beta:", grid.best_score_)

    # Return best model
    return grid.best_estimator_

def predict(model, X, y):
# def predict(model, X_train, y_train, X_val, y_val, X_test , y_test):
    # probs = model.predict_proba(X)
    # preds = model.predict(X)

    # probs = model.predict_proba(X)[:,1]
    probs = model.predict_proba(X)[:, 1]

    # best_threshold = 0.5
    # best_score = -1

    # for threshold in np.arange(0.01, 1.00, 0.01):
    #     preds = (probs >= threshold).astype(int)
    #     score = fbeta_score(y_val, preds, beta=3)

    #     if score > best_score:
    #         best_score = score
    #         best_threshold = threshold

    # print(best_threshold)
    # print(best_score)

    # # retrain on train + validation
    # X_train = pd.concat([X_train, X_val], ignore_index=True)
    # y_train = pd.concat([y_train, y_val], ignore_index=True)

    # final_model = load_classifier(X_train, y_train)

    # # evaluate on test
    # probs = final_model.predict_proba(X_test)[:, 1]
    # preds = (probs >= best_threshold).astype(int)

    preds = (probs >= 0.3).astype(int)

    y_true = y_test

    false_mask = preds != y_true

    false_samples = X_test[false_mask]
    false_probs = probs[false_mask]
    false_preds = preds[false_mask]
    false_true = y_true[false_mask]

    confidence = np.maximum(false_probs, 1 - false_probs)

    stats = {
        "probs": false_probs,
        "preds": false_preds,
        "true": false_true,
        "confidence": confidence,
    }
    # print(f"Misclassified samples: {len(false_true)}")
    # print(stats["confidence"][:10])

    for sample, pred, true, conf in zip(
        false_samples,
        false_preds,
        false_true,
        confidence,
    ):
        print(f"Text      : {sample}")
        print(f"Predicted : {pred}")
        print(f"True      : {true}")
        print(f"Confidence: {conf:.3f}")
        print("-" * 80)

    return preds, stats

print("Ready!")

print("Preparing data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
X_train = pd.concat([X_train, X_val], ignore_index=True)
y_train = pd.concat([y_train, y_val], ignore_index=True)

print("Loading classifier...")
model = load_classifier(X_train, y_train)

print("Making predictions...")
start = time.perf_counter()
predictions, stats = predict(model, X_test, y_test)
end = time.perf_counter()
total_latency = end - start

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

