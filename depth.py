import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import time


def preprocess(df):
    df.drop(['id'], axis=1)
    for i in range(len(df)):
        if df.at[i, 'Previously_Insured'] == 0 and df.at[i, 'Vehicle_Damage'] == 'Yes':
            df.at[i, 'flag'] = 1
        else:
            df.at[i, 'flag'] = 0
    cdat = []
    ndat = []
    for i, c in enumerate(df.dtypes):
            if c == 'object':
                cdat.append(df.iloc[:, i])
            else:
                ndat.append(df.iloc[:, i])

    cdat = pd.DataFrame(cdat).transpose()
    ndat = pd.DataFrame(ndat).transpose()

    le = LabelEncoder()
    for i in cdat:
        cdat[i] = le.fit_transform(cdat[i])
    df = pd.concat([cdat, ndat], axis=1)

    return df


def run_cross_validation_on_trees(X, y, tree_depths, scoring='roc_auc'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth, class_weight='balanced')
        cv_scores = cross_val_score(
            tree_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores


def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(depths, cv_scores_mean, '-o',
            label='mean cross-validation score', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std,
                    cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train score', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('AUC Score', fontsize=14)
    ax.set_ylim(0.5, 1)
    ax.set_xticks(depths)
    ax.legend()
    plt.show()
    

def ti(t1, t2):
    if t2-t1 > 60:
        m = int((t2-t1)//60)
        s = (t2-t1) % 60
        if m > 1:
            print("Time required:", m, "minutes %.3f" % s, "seconds")
        else:
            print("Time required:", m, "minute %.3f" % s, "seconds")
    else:
        print("Time required:%.3f" % (t2-t1), "seconds")


df = pd.read_csv('train.csv')

df0 = preprocess(df)

X = df0.drop(['Response'], axis=1)
y = df0['Response']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

t1=time.time()
sm_tree_depths = range(1, 30)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(
    X_train, y_train, sm_tree_depths)
t2=time.time()

ti(t1,t2)
depth0 = sm_cv_scores_mean.argmax()+1

print(sm_cv_scores_mean)
print("Best tree depth(after cross validation):", depth0)

plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std,
                               sm_accuracy_scores, 'Accuracy per decision tree depth on training data')
