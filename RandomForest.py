## Digit data from scikit-learn
# Predict data using random forest classifier
# cross validate data with K-fold train-test split
from sklearn.datasets import load_digits
digits = load_digits()

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# K-fold train-test split
from sklearn.model_selection import KFold
kf = KFold(n_splits = 4)
accuracy_score_list = []
oob_score_list = []
f1_score_list = []

for i in range(1,11):
    for train_set, test_set in kf.split(digits.data):
        x_train = digits.data[train_set]
        x_test = digits.data[test_set]
        y_train = digits.target[train_set]
        y_test = digits.target[test_set]

        clf = RandomForestClassifier(n_estimators = 100, max_depth = i, random_state = 0, oob_score = True)
        clf.fit(x_train, y_train)
        accuracy_score_list.append(clf.score(x_test, y_test))
        oob_score_list.append(clf.oob_score_)
        f1_score_list.append(f1_score(y_test, clf.predict(x_test), average = 'micro'))
# The list contains the 40 scores (4 scores from each k = 4 split and 10 from max depths = 1, 2,...., 10)
accuracy_avg = [sum(accuracy_score_list[i:i+4])/4 for i in range(0, 40, 4)]
oob_score_avg = [sum(oob_score_list[i:i+4])/4 for i in range(0, 40, 4)]
f1_score_avg = [sum(f1_score_list[i:i+4])/4 for i in range(0, 40, 4)]

plt.plot(accuracy_avg, label = 'accuracy')
plt.plot(oob_score_avg, label = 'oob score')
plt.plot(f1_score_avg, label = 'f1 score')
plt.xlabel('max depth')
plt.ylabel('score')
plt.legend()
plt.show()
# We can see that the scores all saturates as the max depth increases 
