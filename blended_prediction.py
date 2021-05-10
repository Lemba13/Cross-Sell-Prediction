import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,RandomForestClassifier
from sklearn import model_selection
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def preprocess(df):
    df=df.drop(['id'],axis=1)
    sc = StandardScaler()
    df['Age'] = pd.DataFrame(sc.fit_transform(df['Age'].values.reshape(-1, 1)))
    df['Annual_Premium'] = pd.DataFrame(sc.fit_transform(df['Annual_Premium'].values.reshape(-1, 1)))

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

    df['Driving_License'] = df['Driving_License'].astype(int)
    df['Region_Code'] = df['Region_Code'].astype(int)
    df['Previously_Insured'] = df['Previously_Insured'].astype(int)
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype(int)
    df['Vintage'] = (df['Vintage']//30).astype(int)

    return df


def loss(y_true, x, model,model2,model1, ret=False):
    y_pred = model.predict(x)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    roc = roc_auc_score(y_true, ((model.predict_proba(x)[:, 1])+(4*model2.predict_proba(x)[:, 1])+model1.predict_proba(x)[:, 1])/6)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if ret:
        return pre, rec, roc, acc, f1
    else:
        print('  precision: %.5f\n  recall: %.5f\n  roc_auc: %.8f\n  accuracy: %.5f\n  f1: %.5f ' %(pre, rec, roc, acc, f1))


cat_col = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel', 'Vintage']
df=pd.read_csv('train.csv')

df0=preprocess(df)

X = df0.drop(['Response'], axis=1)
y = df0['Response']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

model4 = LGBMClassifier(boosting_type='gbdt', n_estimators=376, max_depth=10, learning_rate=0.04, objective='binary', metric='auc',
                        is_unbalance=True,colsample_bytree=0.5, reg_lambda=2, reg_alpha=2, random_state=42, n_jobs=-1)
model4.fit(X_train, y_train)

model1 = XGBClassifier(learning_rate=0.1,
                       n_estimators=1000,
                       max_depth=4,
                       min_child_weight=1,
                       gamma=0,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       objective='binary:logistic',
                       scale_pos_weight=1,
                       seed=27)
model1.fit(X_train, y_train)

model = CatBoostClassifier()
model = model.fit(X_train, y_train, cat_features=cat_col,eval_set=(X_test, y_test), early_stopping_rounds=30)

loss(y_test, X_test, model4,model,model1)

df_test = pd.read_csv('test.csv')
dft = preprocess(df_test)

x_final = dft
y_final = ((model4.predict_proba(x_final)[:, 1])+(4*model.predict_proba(x_final)[:, 1])+model1.predict_proba(x_final)[:, 1])/6

submission = pd.DataFrame()
submission['id'] = df_test['id']
submission['Response'] = y_final
submission.to_csv('final_submission.csv', header=True, index=False)
