import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# -----------------------
# Load data
# -----------------------
xtr = pd.read_csv('data_readinglevel/x_train.csv')
ytr = pd.read_csv('data_readinglevel/y_train.csv')
xts = pd.read_csv('data_readinglevel/x_test.csv')

X_text_all = xtr['text'].astype(str).str.lower().values
y_all = (ytr['Coarse Label'] == 'Key Stage 4-5').astype(int).values

# -----------------------
# Hyperparameters / CV setup
# -----------------------
C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
skf = StratifiedKFold(n_splits=5)

# Storage for plotting & selection
cv_results = {C: {'train_auc': [], 'val_auc': [], 'train_acc': [], 'val_acc': [], 'vocab_sizes': []}
              for C in C_values}
summary_rows = []

print("\nPerforming 5-Fold Cross Validation...")
for C in C_values:
    print(f"=== Testing C = {C} ===")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text_all, y_all), start=1):
        X_train_text = X_text_all[train_idx]
        X_val_text   = X_text_all[val_idx]
        y_train      = y_all[train_idx]
        y_val        = y_all[val_idx]

        # Vectorizer fit on TRAIN only
        vectorizer = CountVectorizer(
            analyzer='char_wb',
            max_features=80000,
            min_df=5,
            max_df=0.9,
            binary=True,
            ngram_range=(3, 6)
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_val   = vectorizer.transform(X_val_text)

        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(X_train, y_train)

        # Train metrics
        y_train_proba = clf.predict_proba(X_train)[:, 1]
        y_train_pred  = (y_train_proba >= 0.5).astype(int)
        train_auc = roc_auc_score(y_train, y_train_proba)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Val metrics
        y_val_proba = clf.predict_proba(X_val)[:, 1]
        y_val_pred  = (y_val_proba >= 0.5).astype(int)
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_acc = accuracy_score(y_val, y_val_pred)

        # Store
        cv_results[C]['train_auc'].append(train_auc)
        cv_results[C]['val_auc'].append(val_auc)
        cv_results[C]['train_acc'].append(train_acc)
        cv_results[C]['val_acc'].append(val_acc)
        cv_results[C]['vocab_sizes'].append(len(vectorizer.vocabulary_))

    mean_auc = np.mean(cv_results[C]['val_auc'])
    mean_acc = np.mean(cv_results[C]['val_acc'])
    print(f"Mean AUC for C={C}: {mean_auc:.4f}")
    print(f"Mean Accuracy for C={C}: {mean_acc:.4f}")
    print(f"Vocabulary size range across folds: "
          f"{min(cv_results[C]['vocab_sizes'])}-{max(cv_results[C]['vocab_sizes'])}\n")

    summary_rows.append({
        'C': C,
        'train_auc_mean': np.mean(cv_results[C]['train_auc']),
        'train_auc_std':  np.std(cv_results[C]['train_auc']),
        'val_auc_mean':   np.mean(cv_results[C]['val_auc']),
        'val_auc_std':    np.std(cv_results[C]['val_auc']),
        'train_acc_mean': np.mean(cv_results[C]['train_acc']),
        'train_acc_std':  np.std(cv_results[C]['train_acc']),
        'val_acc_mean':   np.mean(cv_results[C]['val_acc']),
        'val_acc_std':    np.std(cv_results[C]['val_acc']),
        'vocab_min':      np.min(cv_results[C]['vocab_sizes']),
        'vocab_max':      np.max(cv_results[C]['vocab_sizes'])
    })

# Pick best C by mean validation AUC
summary_df = pd.DataFrame(summary_rows).sort_values('C')
best_row = summary_df.loc[summary_df['val_auc_mean'].idxmax()]
best_C = float(best_row['C'])
print(f"Best C: {best_C} (Val AUC mean={best_row['val_auc_mean']:.4f})")

# -----------------------
# Plot: Train vs Val AUC with fold dots and ±1 SD
# -----------------------
plt.figure(figsize=(8, 5))

# Per-fold dots (scatter)
rng = np.random.default_rng(0)
jitter_scale = 0.03
for C in C_values:
    # small multiplicative jitter so dots don't overlap perfectly
    x_train = C * (1 + rng.normal(0, jitter_scale, size=len(cv_results[C]['train_auc'])))
    x_val   = C * (1 + rng.normal(0, jitter_scale, size=len(cv_results[C]['val_auc'])))
    plt.scatter(x_train, cv_results[C]['train_auc'], marker='o', alpha=0.6, label=None)
    plt.scatter(x_val,   cv_results[C]['val_auc'],   marker='s', alpha=0.6, label=None)

# Mean lines
train_means = [np.mean(cv_results[C]['train_auc']) for C in C_values]
val_means   = [np.mean(cv_results[C]['val_auc'])   for C in C_values]
train_stds  = [np.std(cv_results[C]['train_auc'])  for C in C_values]
val_stds    = [np.std(cv_results[C]['val_auc'])    for C in C_values]

plt.plot(C_values, train_means, '-o', label='Train AUC (mean)')
plt.plot(C_values, val_means,   '-s', label='Validation AUC (mean)')

# ±1 SD error bars
plt.errorbar(C_values, train_means, yerr=train_stds, fmt='none', capsize=3, alpha=0.8)
plt.errorbar(C_values, val_means,   yerr=val_stds,   fmt='none', capsize=3, alpha=0.8)

plt.xscale('log')
plt.xlabel('C (Inverse Regularization, log scale)')
plt.ylabel('Held-out AUC')
plt.title('5-Fold CV: Training vs Validation Performance vs Complexity (C)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.figtext(
    0.5, -0.08,
    'Caption: Each dot is one fold; lines show per-C means with ±1 SD. '
    'Underfitting at very small C (strong regularization); '
    'overfitting suggested when Train>Val gap widens at large C.',
    wrap=True, ha='center', va='top', fontsize=9
)
plt.tight_layout()
plt.savefig('complexity_curve_auc.png', dpi=200, bbox_inches='tight')
plt.show()

# Also save the numeric summary
summary_df.to_csv('cv_summary_auc_acc.csv', index=False)

# -----------------------
# Train final model on ALL training data using best C, predict test
# -----------------------
final_vectorizer = CountVectorizer(
    analyzer='char_wb',
    max_features=80000,
    min_df=5,
    max_df=0.9,
    binary=True,
    ngram_range=(3, 6)
)
X_all = final_vectorizer.fit_transform(X_text_all)
clf_final = LogisticRegression(max_iter=2000, C=best_C)
clf_final.fit(X_all, y_all)

X_test = final_vectorizer.transform(xts['text'].astype(str).str.lower().values)
yproba_test = clf_final.predict_proba(X_test)[:, 1]
np.savetxt('yproba1_test.txt', yproba_test, fmt='%.6f')

print("\nSaved:")
print(" - complexity_curve_auc.png (figure)")
print(" - cv_summary_auc_acc.csv (per-C means/stds)")
print(" - yproba1_test.txt (test probabilities for leaderboard)")
