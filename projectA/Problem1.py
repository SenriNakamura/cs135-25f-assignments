import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#make tweaks to this later on
vectorizer = CountVectorizer(
    max_features=10000,   # vocabulary of top 10k words
    min_df=5,             # appear in at least 5 docs
    max_df=0.5,           # appear in <50% of docs
    binary=False,         # keep word counts
    stop_words='english'  # drop common English stopwords
)

xtr = pd.read_csv('data_readinglevel/x_train.csv')
ytr = pd.read_csv('data_readinglevel/y_train.csv')
xts = pd.read_csv('data_readinglevel/x_test.csv')
X_text1   = xtr['text'].astype(str).str.lower()
authors  = xtr['author'].astype(str)
y        = (ytr['Coarse Label'] == 'Key Stage 4-5').astype(int).values
X_test_t = xts['text'].astype(str)
X   = vectorizer.fit_transform(X_text1)
x_train, x_val , y_train, y_val = train_test_split(X,y,test_size=.2,random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train)
 
y_pred = clf.predict_proba(x_val)[:, 1]  # probabilities for class 1 (Key stage 4-5)
auc = roc_auc_score(y_val, y_pred)

print(f" Validation AUC: {auc:.4f}")

