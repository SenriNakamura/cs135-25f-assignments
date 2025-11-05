import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

RNG = check_random_state(42)

# ---------------- Helpers ----------------
def load_embeddings(path):
    """Load embeddings from .npz/.npy. For .npz, prefer key 'embeddings' else first 2D array."""
    obj = np.load(path)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = obj.files
        if 'embeddings' in keys:
            arr = obj['embeddings']
        else:
            arr = None
            for k in keys:
                if obj[k].ndim >= 2:
                    arr = obj[k]; break
            if arr is None:
                arr = obj[keys[0]]
    else:
        arr = obj
    return np.asarray(arr)

def build_dense(X_df, bert_mat, fit_objs=None):
    """Return concatenated dense features: [standardized BERT || standardized numeric]."""
    num_cols = [c for c in X_df.columns if c not in ["author","title","passage_id","text"]]
    X_num = X_df[num_cols].values.astype(float)

    if fit_objs is None:
        imp = SimpleImputer(strategy="median").fit(X_num)
        numZ = StandardScaler().fit_transform(imp.transform(X_num))
        bertZ = StandardScaler().fit_transform(bert_mat)
        return np.hstack([bertZ, numZ]), {"imp":imp, "num_scaler":StandardScaler().fit(imp.transform(X_num)),
                                          "bert_scaler":StandardScaler().fit(bert_mat)}
    else:
        imp = fit_objs["imp"]; num_scaler = fit_objs["num_scaler"]; bert_scaler = fit_objs["bert_scaler"]
        X_num_imp = imp.transform(X_num)
        numZ = num_scaler.transform(X_num_imp)
        bertZ = bert_scaler.transform(bert_mat)
        return np.hstack([bertZ, numZ]), fit_objs

def logit_clip(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

# ---------------- Load data ----------------
xtr = pd.read_csv('data_readinglevel/x_train.csv')
ytr = pd.read_csv('data_readinglevel/y_train.csv')
xts = pd.read_csv('data_readinglevel/x_test.csv')
y = (ytr['Coarse Label'] == 'Key Stage 4-5').astype(int).values
authors = xtr['author'].astype(str).values

Xb_tr = load_embeddings('data_readinglevel/x_train_bert_embeddings.npz')
Xb_ts = load_embeddings('data_readinglevel/x_test_bert_embeddings.npz')
print(f"BERT train: {Xb_tr.shape}  |  BERT test: {Xb_ts.shape}")

# ---------------- CV setup (author-grouped) ----------------
gkf = GroupKFold(n_splits=5)

# ---------------- Model A: TF-IDF + Linear SVM (calibrated) ----------------
tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=5, max_df=0.6, sublinear_tf=True)
Cs_svm = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

# ---------------- Model B: MLP on [BERT || numeric] + Platt calibration ----------------
mlp_grid = {
    "hidden":   [256, 384, 512, 768],
    "dropout":  [0.1, 0.2, 0.3, 0.5],
    "alpha":    [1e-5, 1e-4],
    "lr":       [1e-3, 3e-4],
    "weight_decay": [0.0, 1e-5]  # via alpha in sklearn MLP is L2; weight_decay only for torch; we simulate via alpha
}

def fit_mlp(Xtr, Ytr, Xva, Yva, hidden, dropout, alpha, lr):
    # scikit's MLP doesn't expose dropout directly; emulate via early_stopping + alpha + capacity control.
    mlp = MLPClassifier(hidden_layer_sizes=(hidden, max(128, hidden//2)),
                        activation='relu', solver='adam',
                        alpha=alpha, learning_rate_init=lr,
                        max_iter=500, random_state=42,
                        early_stopping=True, n_iter_no_change=15, validation_fraction=0.1)
    mlp.fit(Xtr, Ytr)
    # Platt calibration on validation using logit of predicted probs
    p_tr = mlp.predict_proba(Xtr)[:,1]
    p_va = mlp.predict_proba(Xva)[:,1]
    platt = LogisticRegression(max_iter=1000, solver="lbfgs")
    platt.fit(logit_clip(p_tr).reshape(-1,1), Ytr)
    p_va_cal = platt.predict_proba(logit_clip(p_va).reshape(-1,1))[:,1]
    auc = roc_auc_score(Yva, p_va_cal)
    return mlp, platt, auc, p_va_cal

# ---------------- Cross-validated search & ensemble ----------------
records = []
best_svm_Cs = []
best_mlp_cfgs = []
fold_weights = []
fold_idx_iter = gkf.split(xtr, y, authors)

for f, (tr_idx, va_idx) in enumerate(fold_idx_iter, start=1):
    Xtr_df, Xva_df = xtr.iloc[tr_idx], xtr.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # A) TF-IDF + LinearSVC (with probability via calibration)
    tfidf_fit = tfidf.fit(Xtr_df["text"].astype(str))
    A_tr = tfidf_fit.transform(Xtr_df["text"].astype(str))
    A_va = tfidf_fit.transform(Xva_df["text"].astype(str))

    best_auc_A, best_C, best_proba_A = -1.0, None, None
    for C in Cs_svm:
        base = LinearSVC(C=C, max_iter=5000)
        clfA = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clfA.fit(A_tr, y_tr)
        p_va = clfA.predict_proba(A_va)[:,1]
        aucA = roc_auc_score(y_va, p_va)
        if aucA > best_auc_A:
            best_auc_A, best_C, best_proba_A, best_clfA = aucA, C, p_va, clfA
    best_svm_Cs.append(best_C)

    # B) Dense features + MLP (grid)
    B_tr, fits = build_dense(Xtr_df, Xb_tr[tr_idx], fit_objs=None)
    B_va, _    = build_dense(Xva_df, Xb_tr[va_idx], fit_objs=fits)

    best_auc_B, best_cfg, best_proba_B, best_mlp, best_platt = -1.0, None, None, None, None
    # compact grid: sample a subset for speed
    cfg_list = []
    for h in mlp_grid["hidden"]:
        for dr in mlp_grid["dropout"]:
            for a in mlp_grid["alpha"]:
                for lr in mlp_grid["lr"]:
                    cfg_list.append((h, dr, a, lr))
    # shuffle and try top 12 configs for speed; increase if you want
    RNG.shuffle(cfg_list)
    for (h, dr, a, lr) in cfg_list[:12]:
        mlp, platt, aucB, p_va_cal = fit_mlp(B_tr, y_tr, B_va, y_va, hidden=h, dropout=dr, alpha=a, lr=lr)
        if aucB > best_auc_B:
            best_auc_B, best_cfg, best_proba_B, best_mlp, best_platt = aucB, (h, dr, a, lr), p_va_cal, mlp, platt
    best_mlp_cfgs.append(best_cfg)

    # C) Simple weight for ensemble on this fold
    best_auc_ens, best_w = -1.0, 0.5
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p_ens = w*best_proba_A + (1-w)*best_proba_B
        aucE = roc_auc_score(y_va, p_ens)
        if aucE > best_auc_ens:
            best_auc_ens, best_w = aucE, w
    fold_weights.append(best_w)

    records.append({
        "fold": f,
        "A_auc": best_auc_A,
        "B_auc": best_auc_B,
        "ENS_auc": best_auc_ens,
        "svm_C": best_C,
        "mlp_cfg": best_cfg,
        "w": best_w
    })
    print(f"[Fold {f}]  TFIDF+SVM AUC={best_auc_A:.4f}  |  MLP AUC={best_auc_B:.4f}  |  ENS AUC={best_auc_ens:.4f}  (w={best_w})")

df_rec = pd.DataFrame(records)
print("\nCV Summary (author-grouped 5-fold):")
print(df_rec[["fold","A_auc","B_auc","ENS_auc","svm_C","mlp_cfg","w"]])
print("\nMean AUROC — A: %.4f, B: %.4f, ENS: %.4f" % (df_rec["A_auc"].mean(), df_rec["B_auc"].mean(), df_rec["ENS_auc"].mean()))

# Choose final hyperparams (by mean across folds)
final_C = df_rec.groupby("svm_C")["A_auc"].mean().idxmax()
final_mlp_cfg = (df_rec["mlp_cfg"].value_counts().index[0])  # most frequent best
final_w = np.median(fold_weights)  # robust central tendency

print(f"\nSelected final SVM C = {final_C}")
print(f"Selected final MLP cfg (hidden, dropout, alpha, lr) = {final_mlp_cfg}")
print(f"Selected ensemble weight w (A vs B) = {final_w}")

# ---------------- Fit final on FULL train and predict test ----------------
# Final A: TF-IDF + SVM (calibrated)
tfidf_full = tfidf.fit(xtr["text"].astype(str))
A_full_tr = tfidf_full.transform(xtr["text"].astype(str))
A_full_ts = tfidf_full.transform(xts["text"].astype(str))
clfA_full = CalibratedClassifierCV(LinearSVC(C=final_C, max_iter=5000), method="sigmoid", cv=5)
clfA_full.fit(A_full_tr, y)
pA_ts = clfA_full.predict_proba(A_full_ts)[:,1]

# Final B: Dense + MLP with a small calibration split to avoid leakage
B_full_tr, fits_full = build_dense(xtr, Xb_tr, fit_objs=None)
B_full_ts, _         = build_dense(xts, Xb_ts, fit_objs=fits_full)

n = B_full_tr.shape[0]
idx = np.arange(n); RNG.shuffle(idx)
cut = int(0.9*n)
tr_idx, cal_idx = idx[:cut], idx[cut:]

h, dr, a, lr = final_mlp_cfg
mlp_full = MLPClassifier(hidden_layer_sizes=(h, max(128, h//2)),
                         activation='relu', solver='adam',
                         alpha=a, learning_rate_init=lr,
                         max_iter=800, random_state=42,
                         early_stopping=True, n_iter_no_change=20, validation_fraction=0.1)
mlp_full.fit(B_full_tr[tr_idx], y[tr_idx])

# Platt calibration on held-out 10% of train
p_tr_cal = mlp_full.predict_proba(B_full_tr[cal_idx])[:,1]
platt_full = LogisticRegression(max_iter=1000, solver="lbfgs")
platt_full.fit(logit_clip(p_tr_cal).reshape(-1,1), y[cal_idx])

# Predict test with calibrated MLP
pB_ts_raw = mlp_full.predict_proba(B_full_ts)[:,1]
pB_ts = platt_full.predict_proba(logit_clip(pB_ts_raw).reshape(-1,1))[:,1]

# Ensemble and save
p_ts = final_w*pA_ts + (1-final_w)*pB_ts
np.savetxt("yproba2_test.txt", p_ts, fmt="%.6f")
print("\nSaved yproba2_test.txt (ensemble).  Done.")
