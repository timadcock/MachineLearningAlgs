from ML import Perceptron, LinearRegression, DecisionStump, LogisticalRegression, Threshold, NN, SVM, plot_decision_regions, plot_regression_line, plot_scatter
import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Data
X = df.iloc[0:25, [0,1,2,3]].values
X = np.concatenate((X,df.iloc[50:75, [0,1,2,3]].values))
X = np.concatenate((X,df.iloc[100:125, [0,1,2,3]].values))

Y = df.iloc[0:25, 4].values
Y = np.concatenate((Y,df.iloc[50:75, 4].values))
Y = np.concatenate((Y,df.iloc[100:125, 4].values))
Y = np.where(Y == 'Iris-virginica', 1, -1)

TX = df.iloc[25:50, [0,1,2,3]].values
TX = np.concatenate((TX,df.iloc[75:100, [0,1,2,3]].values))
TX = np.concatenate((TX,df.iloc[125:150, [0,1,2,3]].values))

TY = df.iloc[25:50, 4].values
TY = np.concatenate((TY,df.iloc[50:75, 4].values))
TY = np.concatenate((TY,df.iloc[100:125, 4].values))
TY = np.where(TY == 'Iris-virginica', 1, -1)

#SVM
svmX = df.iloc[0:50, [0,2]].values
svmX = np.concatenate((svmX,df.iloc[100:150, [0,2]].values))

svmY = df.iloc[0:50, 4].values
svmY = np.concatenate((svmY,df.iloc[100:150, 4].values))
svmY = np.where(svmY == 'Iris-setosa', 1, -1)

pn = SVM()
pn.fit(svmX, svmY)
preds = pn.predict(svmX)

err = 0
for i in range(len(svmY)):
    if svmY[i] != preds[i]:
        err += 1

print("SVM gave an error of", err)
print("Upper weights: ", pn.upper)
print("Optimized weights: ", pn.weight)
print("Lower weights: ", pn.lower)

plot_decision_regions(svmX,svmY,pn,('petal length','sepal length'))

# k-NN
pn = NN(k=2)
pn.fit(X,Y)
preds = pn.predict(TX)

err = 0
for i in range(len(preds)):
    print("Closest to 2 Neighbors\n", preds[i][0], "\nwith average label ", preds[i][1], "Actual label ", TY[i],"\n")
    if preds[i][1] != TY[i]:
        err += 1
print("Nearest Neighbor error: ", err)

#Threshold
pn = Threshold()
pn.fit(X,Y)
preds = pn.predict(TX)

err = 0
for i in range(len(preds)):
    if preds[i] != TY[i]:
        err += 1
print("Threshold error using Perceptron: ", err)

#Logisitcal Regression
pn = LogisticalRegression()
pn.fit(X,Y)
probs = pn.predict(TX)
err = 0
for i in range(len(probs)):
    print(probs[i])
    if np.where(probs[i] >= .5,1,-1) != TY[i]:
        err += 1

print("Logistical Regression Error: ", err)
plot_scatter(TY, probs, "Label", "Probability of 1")

#Decision Stump
pn = DecisionStump()
pn.fit(X,Y)
pred = pn.predict(TX)
cnt = 0
for i in range(len(pred)):
    if pred[i] != TY[i]:
        cnt += 1

print("Threshold: ", pn.theta)
print("Variable: ", pn.j)
print("Classification: ", pn.s)
print("Errors: ", cnt)

#Linear Regression
linY = df.iloc[0:150, 4].values
linX = df.iloc[0:150, [0,1]].values

pn = LinearRegression()
pn.fit(linX[:,0:1],linX[:,-1])
print("Linear Regression weights: ", pn.weight)
plot_regression_line(linX[:,0:1],linX[:,-1],linY,pn,"Sepal Length","Sepal Width")

#Perceptron
perX = df.iloc[0:50, [0,2]].values
perX = np.concatenate((perX,df.iloc[100:150, [0,2]].values))

perY = df.iloc[0:50, 4].values
perY = np.concatenate((perY,df.iloc[100:150, 4].values))
perY = np.where(perY == 'Iris-setosa', 1, -1)

pn = Perceptron(.01, 100)

pn.fit(perX, perY)

plot_decision_regions(perX,perY,pn,('petal length','sepal length'))
