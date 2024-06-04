from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)
# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X):
 # split data
 X_train, X_test = X[train_ix, :], X[test_ix, :]
 y_train, y_test = y[train_ix], y[test_ix]
 # configure the cross-validation procedure
 cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
 # define the model
 model = RandomForestClassifier(random_state=1)
 # define search space
 space = dict()
 space['n_estimators'] = [10, 100, 500]
 space['max_features'] = [2, 4, 6]
 # define search
 search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
 print('searching...')
 # execute search
 result = search.fit(X_train, y_train)
 # get the best performing model fit on the whole training set
 best_model = result.best_estimator_
 # evaluate model on the hold out dataset
 yhat = best_model.predict(X_test)
 # evaluate the model
 acc = accuracy_score(y_test, yhat)
 # store the result
 outer_results.append(acc)
 # report progress
 print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))