from sklearn.model_selection import *
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3)

# lr = LogisticRegression().fit(x_train, y_train)
# print(f'LR score is {lr.score(x_test,y_test)}')

# SVM = SVC().fit(x_train, y_train)
# print(f'SVM score is {SVM.score(x_test,y_test)}')

# rf = RandomForestClassifier().fit(x_train, y_train)
# print(f'rf score is {rf.score(x_test,y_test)}')


def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


folds = StratifiedKFold(n_splits=3)

# scores_logistic = []
# scores_svm = []
# scores_rf = []

# for train_index, test_index in folds.split(digits.data, digits.target):
#     X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
#         digits.target[train_index], digits.target[test_index]
#     scores_logistic.append(get_score(LogisticRegression(
#         solver='liblinear', multi_class='ovr'), X_train, X_test, y_train, y_test))
#     scores_svm.append(get_score(SVC(gamma='auto'),
#                       X_train, X_test, y_train, y_test))
#     scores_rf.append(get_score(RandomForestClassifier(
#         n_estimators=40), X_train, X_test, y_train, y_test))

# print(scores_logistic)
# print(scores_svm)
# print(scores_rf)

print(cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target,cv=3))
print(cross_val_score(SVC(gamma='auto'), digits.data, digits.target,cv=3))
print(cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3))