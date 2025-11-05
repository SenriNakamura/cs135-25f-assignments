import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Helpers ----------
def load_embeddings(path):
    """Load embeddings from .npz or .npy. For .npz, try 'embeddings' or first 2D array."""
    obj = np.load(path)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = obj.files
        # Prefer a key named 'embeddings' if present
        if 'embeddings' in keys:
            arr = obj['embeddings']
        else:
            # fall back to first 2D array; else first key
            arr = None
            for k in keys:
                if obj[k].ndim >= 2:
                    arr = obj[k]
                    break
            if arr is None:
                arr = obj[keys[0]]
    else:
        # .npy returns an ndarray directly
        arr = obj
    return np.asarray(arr)

# === Load CSVs ===
xtr = pd.read_csv('data_readinglevel/x_train.csv')
ytr = pd.read_csv('data_readinglevel/y_train.csv')
xts = pd.read_csv('data_readinglevel/x_test.csv')

# Binary labels
y = (ytr['Coarse Label'] == 'Key Stage 4-5').astype(int).values

# === Load BERT embeddings ===
X_train = load_embeddings('data_readinglevel/x_train_bert_embeddings.npz')
X_test  = load_embeddings('data_readinglevel/x_test_bert_embeddings.npz')

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape:     {X_test.shape}")

# Standardize features (fit on train only)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

# Define MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation='tanh',
    solver='adam',
    alpha=1e-3,            # L2; you can tune this 1e-4
    learning_rate_init=1e-3,
    max_iter=100,          # 50 can underfit; 200 is safer
    random_state=42,
    early_stopping=True,   # helps generalization
    n_iter_no_change=10,
    validation_fraction=0.1
)

# === Cross-validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, acc_scores = [], []

print("\nPerforming 5-Fold Stratified CV with BERT + MLP...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_std, y), start=1):
    X_tr, X_val = X_train_std[train_idx], X_train_std[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    mlp.fit(X_tr, y_tr)
    y_pred_proba = mlp.predict_proba(X_val)[:, 1]
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred_bin)
    auc_scores.append(auc)
    acc_scores.append(acc)

    print(f"Fold {fold}: AUC={auc:.4f}, ACC={acc:.4f}")

print(f"\nMean AUC: {np.mean(auc_scores):.4f}  ± {np.std(auc_scores):.4f}")
print(f"Mean ACC: {np.mean(acc_scores):.4f}  ± {np.std(acc_scores):.4f}")

# === Train final model on full training data ===
mlp.fit(X_train_std, y)
yproba_test = mlp.predict_proba(X_test_std)[:, 1]

# Save probabilities for leaderboard
np.savetxt("yproba2_test.txt", yproba_test, fmt="%.6f")
print("\nFinal model trained and test predictions saved to yproba2_test.txt")
