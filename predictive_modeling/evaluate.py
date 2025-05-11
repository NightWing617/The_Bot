
# evaluate.py

from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    return {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4)
    }
