from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,0.1], random_state=1)
classifier_list = [KNeighborsClassifier,GaussianNB,LogisticRegression,SVC,DecisionTreeClassifier,RandomForestClassifier,AdaBoostClassifier]
def machine_learning(classifier, X, y, **kwargs):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)
    
    probability = kwargs.get('probability', True)
    if classifier == SVC:
        model = classifier(probability = probability)
    else:
        model = classifier()
    model.fit(train_X, train_y)
    
    probs = model.predict_proba(test_X)
    probs = probs[:, 1]
    auc = roc_auc_score(test_y, probs)
    y_pred = model.predict(test_X)
    f1 = f1_score(test_y, y_pred)   
          
    fpr, tpr, thresholds = roc_curve(test_y, probs) 
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].plot(fpr, tpr, marker='.')
    ax[0].set_xlabel('false positive rate')
    ax[0].set_ylabel('true positive rate')
    ax[0].set_title('ROC curve')

    precision, recall, thresholds = precision_recall_curve(test_y, probs)

    ax[1].step(recall, precision, alpha = 0.2)
    ax[1].set_xlabel('recall')
    ax[1].set_ylabel('precision')
    ax[1].set_title('precision-recall curve')    
    plt.suptitle(classifier.__name__)
    return f'the {classifier.__name__} gives auc of {auc} and f1 score of {f1}'

for clf in classifier_list:
    print(machine_learning(clf, X, y))
# Turned out that ROC generally performs well when the data is imbalanced but f1 score tells us that the classifier is not performing well
# Also most classification methods don't do well for imbalanced data with default settings 
