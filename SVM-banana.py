## scikit-learn with SVM
# Banana-shape data from https://github.com/QCB-Collaboratory/W17.MachineLearning/wiki/Day-2 
import numpy as np
import matplotlib.pyplot as plt

banana = np.loadtxt('banana.csv')
targets = banana[:,0]
features = banana[:,[1,2]]

# plot the data with feature 1 or 2 to see how they distribute
plt.plot(features[:,0][targets == 1],features[:,1][targets == 1], 's', color = 'b')
plt.plot(features[:,0][targets == 2],features[:,1][targets == 2], 's', color = 'r')
plt.xlabel('feature 1')
plt.ylabel('feature 2')

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train,X_test,Y_train,Y_test = train_test_split(features, targets, test_size = 0.6)

gamma = [0.01*3**i for i in range(30)]
train_score = []
test_score = []
for g in gamma:
    SVC_model = SVC(gamma = g)
    SVC_model.fit(X_train,Y_train)
    train_score.append(SVC_model.score(X_train,Y_train))
    test_score.append(SVC_model.score(X_test,Y_test))
 
plt.plot(gamma,train_score,color='b',label = 'train')
plt.plot(gamma,test_score,color='r', label = 'test')
plt.xlim(0, 2) # set the x limit at 2 so I can see where the score reaches maximum
plt.legend()
plt.xlabel('gamma')
plt.ylabel('score')
# although the score keeps going up using higher gamma with the training set, the test set allows us to know that the score reaches it max when the gamma is smaller than 0.5
best_score_index = test_score.index(max(test_score))
print('The best gamma is:', gamma[best_score_index-1], ', with the score of:', test_score[best_score_index-1])


# Next, we'll find the best C value
c_list = [0.01*1.13**i for i in range(40)] 

test_score = []
train_score = []
for c in c_list:
    SVC_model = SVC(gamma = 0.27, C = c)
    SVC_model.fit(X_train,Y_train)
    test_score.append(SVC_model.score(X_train,Y_train))
	train_score.append(SVC_model.score(X_test,Y_test))
plt.plot(c_list,train_score,color='b',label = 'train')
plt.plot(c_list,test_score,color='r', label = 'test')
plt.xlim(0, 2) # set the x limit at 2 so I can see where the score reaches maximum
plt.legend()
plt.xlabel('C value')
plt.ylabel('score')
best_score_index = test_score.index(max(test_score))
print('The best C is:', c_list[best_score_index-1], ', with the score of:', test_score[best_score_index-1])

# Make a contour plot to see how the decision boundaries look like
X,Y = np.meshgrid(x,y)
SVC_model = SVC(gamma = 0.27, C = 0.166)
SVC_model.fit(features, targets)
Z = SVC_model.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

cset1 = plt.contourf(X,Y,Z, cmap = plt.get_cmap('binary'))
# Overlap the actual data on top of the decision boundary 
plt.plot(features[:,0][targets == 1],features[:,1][targets == 1], 's', color = 'b')
plt.plot(features[:,0][targets == 2],features[:,1][targets == 2], 's', color = 'r')
plt.xlabel('feature 1')
plt.ylabel('feature 2')

# Alternatively, GridSearchCV is a good method to find the parameters with the best accuracy score
from sklearn.model_selection import GridSearchCV
SVC_model = SVC()
param_range = np.logspace(-3, 3, 7)
print(SVC_model.get_params().keys()) # Parameters that could be tuned
param_grid = [{'C':param_range,
	       'kernel':['linear'],
	       'gamma':param_range},
	       {'C':param_range,
	       'kernel':['rbf'],
	       'gamma':param_range}]
gs = GridSearchCV(estimator = SVC_model,
		 param_grid = param_grid,
		 scoring = 'accuracy',
		 cv = 5,
		 n_jobs = -1)
gs = gs.fit(X_train, Y_train)
print(f'The best parameters are {gs.best_params_} with the accuracy score of {gs.best_score_}')
