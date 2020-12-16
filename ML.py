import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter
        self.errors = []
        self.weight = np.zeros(1)


    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]"""

        #weights: create a weights array of right size
        # and initializeelements to zero
        self.weight = np.zeros(X.shape[1] + 1)

        # Number of misclassifications, creates an array
        # to hold the number of misclassifications
        self.errors = []

        # main loop to fit the data to the labels
        for i in range(self.niter):
            # set iteration error to zero
            # loop over all the objects in X and corresponding y element
            miscal = 0
            for xi, target in zip(X, y):
                # calculate the needed (delta_w) update from previous step
                # delta_w= rate * (target –prediction current object)
                dw = self.rate * (target - self.predict(xi))

                # calculate what the current object will add to the weight
                self.weight[1:] += dw * xi

                # set the bias to be the current delta_w
                self.weight[0] += dw

                # increase the iteration error if delta_w != 0
                if dw != 0:
                    miscal += 1

            # Update the misclassificationarray with # of errors in iteration
            self.errors.append(miscal)

            if miscal == 0:
                return self

        # return self
        return self

    def net_input(self, X):
        """Calculate net input"""
        # return the dot product: X.w+ bias
        if X.shape[0] + 1 != self.weight.shape[0]:
            tmp = []
            for x in X:
                t = np.dot(self.weight[1:], x) + self.weight[0]
                tmp.append(t)
        else:
            tmp = np.dot(self.weight[1:], X) + self.weight[0]

        return np.array(tmp)


    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1,-1)




class LinearRegression():
    def __init__(self):
        self.weight = np.zeros(1)


    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]"""

        """
            solves beta = X^(-1)y through beta = (X^TX)^(-1)y since we can not
            guarntee that X is invertabe I use the pseudo inverse.
            I chose this method since it can be expanded to use multiple vars,
            and it is simple to understand since it looks more like basic algebra.

            I have tried using multiple vars already but I found that how I graph
            them does not look right so I am strictly using this for simple
            linear regression.
        """
        xf = [np.insert(xt,0,1) for xt in X] # insert x_0 = 1
        pinv =  np.linalg.pinv(xf) #pseudo inverse
        self.weight = pinv.dot(y) #dot the psuedo inverse with y

    def predict(self, X):
        """
            This is really just a copy paste from the Perceptrion since it also uses
            a dot product to get a prediction.
            This would look like y = (X)beta.
        """
        if len(X.shape) > 1:
            result = []
            for xi in X:
                pred = self.predict(xi)
                result.append(pred)
        else:
            val = np.array([1, *X])
            result = np.dot(val, self.weight)

        return result


class DecisionStump():
    def __init__(self):
        self.theta = None
        self.j = None
        self.s = 1

    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]"""


        theta = 0
        mine = np.inf
        """
            Go through all the variables availiable ie. X[0] = a,b,c go through
            all a vars in the data then b vars then c.
        """
        for j in range(len(X[0])):
            """
                Since I iterate through all observations anyways I decided not
                to sort the data since it would add atleast another n iterations
            """
            xj = X[:,j]
            min = np.inf
            """
                Go through all observations using that single variable
            """
            for i in range(len(xj)):
                """
                    Go through all observation labels, really there is only -1, 1
                    but just in case there are more.
                """
                for s in np.unique(y):
                    """
                        These will get the theta(threshold) to check to see if it
                        would be the best decision threshold.
                    """
                    if i == 0:
                        theta = (xj[i] + 1)/2.0
                    elif i == len(xj) - 1:
                        theta = (xj[i] - 1)/2.0
                    else:
                        theta = (xj[i] + xj[i-1])/2.0

                    cnt = 0
                    """
                        Check to see if the theta (threshold), j (variable)
                        and k (label) would be a good combination, this really
                        just outputs how many it predicted incorrectly, a call to
                        predict does the same thing with the exception of counting
                        the errors.
                    """
                    for k in range(len(xj)):
                        tmp = s * np.sign(xj[k] - theta )
                        if tmp != y[k] or tmp == 0:
                            cnt += 1

                    """
                        if the the new combination has a lower error make it the
                        new 'winning' combination
                    """
                    if cnt < mine:
                        mine = cnt
                        self.theta = theta
                        self.j = j
                        self.s = s


    def predict(self,X):
        """
            solves s*sign(x_j - θ)
        """
        tmp = (self.s * np.sign(X[:,self.j] - self.theta))
        tmp1 = tmp == 0
        tmp[tmp1] = self.s
        return tmp


class LogisticalRegression():
    def __init__(self, model = Perceptron(.01, 100)):
        self.model = model
        self.weight = None

    def fit(self, X, y):
        '''
        Here I use the Perceptron to get weights for the logistical regression
        '''
        self.model.fit(X, y)
        self.weight = self.model.weight


    def predict(self, X):
        '''
        Since logisitical regression gives probabilities the predict function will use the data given and predict that data using the weights, then solve s/(1+exp(-(b_0 + bxi))) sigmoid function
        '''
        tmp = np.dot(X,self.weight[1:]) + self.weight[0]
        prob = []

        for xi in tmp:
            tmp1 = 1/(1 + np.exp(-(xi)))
            prob.append(tmp1)

        return prob


class Threshold():
    def __init__(self, model = Perceptron()):
        self.model = model
        self.theta = 0
        self.weight = None

    def fit(self, X, y):
        '''
        Threshold will use a given model that has a self.weight variable that represents the weights used in the model. In this library only Peceptron and Logistical and Linear Regressions can be used.
        '''
        self.model.fit(X,y)
        self.weight = self.model.weight
        preds = np.dot(X,self.weight[1:]) + self.weight[0]

        mine = np.inf
        """
            This is a duplicate of the Decision Stump fitting modified for Thresholding.
            Go through all observations using that single variable
        """
        for i in range(len(preds)):
            """
                Go through all observation labels, really there is only -1, 1
                but just in case there are more.
            """
            """
                These will get the theta(threshold) to check to see if it
                would be the best decision threshold.
            """
            if i == 0:
                theta = (preds[i] + 1)/2.0
            elif i == len(preds) - 1:
                theta = (preds[i] - 1)/2.0
            else:
                theta = (preds[i] + preds[i-1])/2.0

            cnt = 0
            """
                Check to see if the theta (threshold), j (variable)
                and k (label) would be a good combination, this really
                just outputs how many it predicted incorrectly, a call to
                predict does the same thing with the exception of counting
                the errors.
            """
            for k in range(len(preds)):
                tmp = np.sign(preds[k] - theta )
                if tmp != y[k] or tmp == 0:
                    cnt += 1

            """
                if the the new combination has a lower error make it the
                new 'winning' combination
            """
            if cnt < mine:
                mine = cnt
                self.theta = theta

    def predict(self,X):
        '''
            This will get a scalar from the weights and the data to be predicted then use the calcualted threshold to determine the label of the element.
        '''
        tmp = np.dot(X,self.weight[1:]) + self.weight[0]

        return np.where(tmp - self.theta >= 0, 1, -1)

class NN():
    def __init__(self, k = 1):
        self.k = k
        self.X = None
        self.y = None

    def fit(self,X,y):
        '''
            Store the X and y values
        '''
        self.X = np.array(X)
        self.y = np.array(y)

    def closest(self,x):
        '''
         Get the k nearest neighbors
        '''
        dis = []

        '''
            This will get all the distances from the given point
        '''
        for e in range(len(self.X)):
            d = np.linalg.norm(x-self.X[e])
            dis.append([d,e])

        '''
            Sort the distance/index array based on distance
        '''
        sort = np.array(sorted(dis))

        return sort[:self.k]

    def predict(self,X):
        '''
            This will get the k nearest neighbors and return them with an estimated label
        '''
        nn = []
        for x in X:
            tmp = self.closest(x)
            labs = self.y[tmp[:,1].astype(int)]
            xs = self.X[tmp[:,1].astype(int)]
            lab = np.where(np.average(labs) >= 0, 1, -1)
            nn.append([xs,lab])

        return nn

class SVM():
    def __init__(self):
        self.weight = None

    def fit(self,X, y):
        pn = Perceptron()
        pn.fit(X,y)

        self.weight = pn.weight#/np.linalg.norm(pn.weight)

        minL = np.inf
        minU = -np.inf

        L = None
        U = None

        for x in X[y == 1]:
            val = np.array([1, *x])
            item = np.linalg.norm(np.dot(val, self.weight))
            if minL >= item:
                minL = item
                L = x

        for x in X[y == -1]:
            val = np.array([1, *x])
            item =  np.dot(val, self.weight)/np.sqrt(np.sum(x ** 2))

            if minU <= item:
                minU = item
                U = x

        upper = self.weight[0] / np.dot(np.array([1, *U]), self.weight)
        lower = self.weight[0] / np.dot(np.array([1, *L]), self.weight)

        self.lower = self.weight.copy()
        self.upper = self.weight.copy()

        self.lower[0] = (upper - lower) / 2
        self.upper[0] = (-upper + lower) / 2
        self.weight[0] = (upper + lower) / 2

    def net_input(self, X):
        """Calculate net input"""
        # return the dot product: X.w+ bias
        if X.shape[0] + 1 != self.weight.shape[0]:
            tmp = []
            for x in X:
                t = np.dot(self.weight[1:], x) + self.weight[0]
                tmp.append(t)
        else:
            tmp = np.dot(self.weight[1:], X) + self.weight[0]
        return np.array(tmp)


    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1,-1)



#Author Dr. Karlsson
#---------------
def plot_decision_regions(X, y, classifier, labels, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:,  0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
        alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(loc='upper left')

    plt.show()
#---------------

def plot_scatter(X, y, xlabel = "X", ylabel = "Y"):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    plt.scatter(x=X, y=y, alpha=0.8)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_regression_line(X, y, label, classifier, xlabel = "X", ylabel = "Y"):
    """
    X is the data to be graphed (only will do first and last variable)
    y is the labels for those points
    w are the b (beta) values calculated from
    """
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    line = []

    preds = classifier.predict(X)

    line = [[x,y] for x,y in zip(X,preds)]

    line = np.array(line)

    for idx, cl in enumerate(np.unique(label)):
        plt.scatter(x=X[label == cl], y=y[label == cl],
        alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

    plt.legend()
    plt.plot(line[:,0],line[:,1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
