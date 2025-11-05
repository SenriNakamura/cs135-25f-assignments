import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Helper: load BERT embeddings ----------
def load_embeddings(path):
    """Load embeddings from .npz or .npy. For .npz, try 'embeddings' or first 2D array."""
    obj = np.load(path)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = obj.files
        if 'embeddings' in keys:
            arr = obj['embeddings']
        else:
            arr = next((obj[k] for k in keys if obj[k].ndim >= 2), obj[keys[0]])
    else:
        arr = obj
    return np.asarray(arr)

# === Load data ===
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

# === Standardize features ===
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

# === Define MLP base model ===
base_mlp = MLPClassifier(
    activation='relu',
    solver='adam',
    random_state=42,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1
)

# === Define hyperparameter grid ===
param_grid = {
    "hidden_layer_sizes": [(256,), (256, 128), (512, 256)],
    "alpha": [1e-5, 1e-4, 1e-3],             # regularization strength
    "learning_rate_init": [1e-4, 1e-3, 5e-3],
    "max_iter": [100, 200, 300]
}

# === Stratified 3-fold CV to save time ===
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# === Define GridSearchCV ===
grid_search = GridSearchCV(
    estimator=base_mlp,
    param_grid=param_grid,
    scoring=make_scorer(roc_auc_score, needs_proba=True),
    cv=cv,
    n_jobs=-1,           # use all CPU cores
    verbose=2
)

# === Fit grid search ===
print("\n Running GridSearchCV for MLP on BERT embeddings...")
grid_search.fit(X_train_std, y)

# === Display results ===
print("\nBest hyperparameters found:")
print(grid_search.best_params_)
print(f"Best mean AUROC: {grid_search.best_score_:.4f}")

# === Retrain final model on all training data ===
best_mlp = grid_search.best_estimator_
best_mlp.fit(X_train_std, y)

# === Generate test predictions ===
yproba_test = best_mlp.predict_proba(X_test_std)[:, 1]

np.savetxt("yproba2_best_test.txt", yproba_test, fmt="%.6f")
print("\nTest predictions saved to yproba2_best_test.txt")