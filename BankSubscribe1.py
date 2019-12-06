import numpy
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import matthews_corrcoef

df = pd.read_csv('./data/train.csv', sep=';')
df_test = pd.read_csv('./data/test.csv', sep=';', names=df.columns)
index_df_test = []
for i in range(df.shape[0], df.shape[0] + df_test.shape[0]):
    index_df_test.append(i)
df_test = df_test.set_index([index_df_test])

# frame = pd.concat([df, df_test])
frame = pd.concat([df])
frame = frame.drop(['month', 'day_of_week'], axis=1)

numerique = [c for c, d in zip(frame.columns, frame.dtypes) if d == numpy.int64]
categories = [c for c in frame.columns if c not in numerique and c not in ["y"]]
num = frame[numerique]
cat = frame[categories]

prep = DictVectorizer()
cat_as_dicts = [dict(r.iteritems()) for _, r in cat.iterrows()]
cat_exp = prep.fit_transform(cat_as_dicts).toarray()
cat_exp_df = pd.DataFrame(cat_exp, columns=prep.feature_names_)

reject = ['default=no', 'education=unknown', 'contact=unknown', 'housing=no', 'job=unknown', 'loan=no',
          'poutcome=unknown']
keep = [c for c in cat_exp_df.columns if c not in reject]
cat_exp_df_nocor = cat_exp_df[keep]
# X = pd.concat([num, cat_exp_df_nocor], axis=1)
X = pd.concat([num, cat_exp_df], axis=1)

target = df["y"]
Y = target.apply(lambda r: (1 if r == "yes" else 0))

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

# from sklearn.ensemble import RandomForestClassifier
#
# clf = RandomForestClassifier(n_estimators=100, criterion='entropy', class_weight={0: 1, 1: 10})
# clf.fit(X_train, Y_train)
# print(matthews_corrcoef(Y_test, clf.predict(X_test)))

from sklearn import svm

# from sklearn.model_selection import GridSearchCV
#
# Cs = numpy.logspace(-6, -1, 10)
# svc = svm.SVC(kernel='linear', class_weight={0: 1, 1: 10})
# clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs))
# clf.fit(X_train, Y_train.ravel())
# print(clf.best_params_)

clf = svm.SVC(kernel='linear', C=0.1, class_weight={0: 1, 1: 10})
clf.fit(X_train, Y_train.ravel())
print(matthews_corrcoef(Y_test, clf.predict(X_test)))


# clf.fit(X[:df.shape[0]], Y[:df.shape[0]].ravel())
#
# # Training data
#
# X_test = X[-df_test.shape[0]:]
# test_Y = clf.predict(X_test)
#
# df_result = pd.DataFrame(test_Y, columns=['Prediction'])
# df_result["Id"] = df_result.index + 1
# df_result.to_csv('./data/SVM1.csv', index=False)
