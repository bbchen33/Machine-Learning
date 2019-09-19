# function for drawing the decision boundaries determined by a machine learning classification
# To visualize the boundary when 2 features are used
def decision_boundaries(X, y, classifier, resolution):
  import matplotlib.pyplot as plt
  import numpy as np
	x1_min, x1_max = X[:,0].min()- 0.5, X[:,0].max()+0.5
	x2_min, x2_max = X[:,1].min()- 0.5, X[:,1].max()+0.5
	
	a, b = np.meshgrid(np.arange(x1_min, x1_max, resolution),
						np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.c_[a.ravel(), b.ravel()])
	
	Z = Z.reshape(a.shape)
	plt.contourf(a, b, Z, alpha = 0.4, cmap = plt.get_cmap('brg'))


# After running the function, one can add the data points on top of it and do plt.show() to show the figure

