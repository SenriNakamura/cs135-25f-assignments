import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
xtr = pd.read_csv('data_readinglevel/x_train.csv')
ytr = pd.read_csv('data_readinglevel/y_train.csv')
xts = pd.read_csv('data_readinglevel/x_test.csv')

X_text = xtr['text'].astype(str).str.lower()
y = (ytr['Coarse Label'] == 'Key Stage 4-5').astype(int).values

# Define vectorizer and classifier
vectorizer = CountVectorizer(
    analyzer='char_wb',
    max_features=80000,
    min_df=5,
    max_df=0.9,
    binary=True,
    ngram_range=(3,6)
)

clf = LogisticRegression(max_iter=2000, C=0.001)

# Cross-validation setup
C_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
skf = StratifiedKFold(n_splits=5)
vocab_sizes = []
auc_scores = []
acc_scores = []

print("\nPerforming 5-Fold Cross Validation...")

results = []

for C in C_values:
    clf = LogisticRegression(max_iter=2000, C=C)
    auc_scores = []
    acc_scores = []
    vocab_sizes = []

    print(f"=== Testing C = {C} ===")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y), start=1):
        X_train_text, X_val_text = X_text.iloc[train_idx], X_text.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit vectorizer on training fold only
        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)

        vocab_size = len(vectorizer.vocabulary_)
        vocab_sizes.append(vocab_size)

        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred_binary)
        auc_scores.append(auc)
        acc_scores.append(acc)

    mean_auc = np.mean(auc_scores)
    mean_acc = np.mean(acc_scores)
    results.append((C, mean_auc, mean_acc))

    print(f"Mean AUC for C={C}: {mean_auc:.4f}")
    print(f"Mean Accuracy for C={C}: {mean_acc:.4f}")
    print(f"Vocabulary size range across folds: {min(vocab_sizes)}-{max(vocab_sizes)}\n")

# Choose Best C val
best_C, best_auc, best_acc = max(results, key=lambda x: x[1])
print(f"Best C: {best_C} (AUC={best_auc:.4f}, ACC={best_acc:.4f})")

# Retrain Final Model on Full Training Data with Best C
vectorizer.fit(X_text)
X_all = vectorizer.transform(X_text)
clf_final = LogisticRegression(max_iter=2000, C=best_C)
clf_final.fit(X_all, y)

X_test = vectorizer.transform(xts['text'].astype(str))
yproba_test = clf_final.predict_proba(X_test)[:, 1]

# Save probs for leaderboard submission
np.savetxt("yproba1_test.txt", yproba_test, fmt="%.6f")

print("\nFinal model trained with best C. Predictions saved to yproba1_test.txt")
cm = confusion_matrix(y_val, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['KS2-3', 'KS4-5'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Best Logistic Regression Classifier")
plt.show()