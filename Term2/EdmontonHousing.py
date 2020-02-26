%matplotlib inline

import numpy as np                               # General math operations
import scipy.io as sio                           # Loads .mat variables
import matplotlib.pyplot as plt                  # Data visualization
from sklearn.linear_model import Perceptron      # Perceptron toolbox
from sklearn.neural_network import MLPRegressor  # MLP toolbox
import seaborn as sns 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn import datasets 
from sklearn.neural_network import MLPClassifier 
from sklearn import preprocessing
from sklearn import linear_model                # Linear models
from sklearn.tree import DecisionTreeRegressor 

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('EdmontonRealEstateData.csv')
sns.distplot(df_train['assessed_value'])


# load the data
iris = datasets.load_iris()
Y = iris.target
X = iris.data

# set up the pandas dataframes 
X_df = pd.DataFrame(X, columns = ['Sepal length','Sepal width', 'Petal length', 'Petal width'] )
Y_df = pd.DataFrame(Y, columns = ['Iris class'])

# this code changes the class labels from numerical values to strings
Y_df = Y_df.replace({
0:'Setosa',
1:'Virginica',
2:'Versicolor'
})

#Joins the two dataframes into a single data frame for ease of use
Z_df = X_df.join(Y_df)

# show the data using seaborn 
sns.set(style='dark', palette= 'deep')
pair = sns.pairplot(Z_df, hue = 'Iris class')
plt.show()

RANDOM_SEED = 6
xTrain, xTest, yTrain, yTest = train_test_split(X_df, Y_df, test_size =0.3,\
                                               random_state=RANDOM_SEED)
#plot the testing data 
test_df = xTest.join(yTest)
# print(test_df.head)
# perceptron training
percep = Perceptron(max_iter = 1000)
percep.fit(xTrain, yTrain)
prediction = percep.predict(xTest)

# print(prediction)
# display the classifiers performance  
prediction_df = pd.DataFrame(prediction, columns=['Predicted Iris class'], index = test_df.index)
# print(prediction_df.head)

prediction_df_index_df = prediction_df.join(xTest)
# print(prediction_df_index_df.head)

pair = sns.pairplot(prediction_df_index_df, hue = 'Predicted Iris class')
#pair_test = sns.pairplot(test_df, hue ='Iris class')
plt.show()

pair_test = sns.pairplot(test_df, hue ='Iris class') #test data from the dataset 

# change the layers, retrain the mlp 
cls = MLPClassifier(solver = 'sgd' ,activation = 'relu' ,  \
                    hidden_layer_sizes = (8,3,), max_iter = 100000)

for i in range(0,5):
    cls.fit(xTrain, yTrain)

mlp_z = cls.predict(xTest)

mlp_z.reshape(-1,1)


cls_df = pd.DataFrame(mlp_z, columns = ["Mlp prediction"], index=xTest.index)

# cls_df_index = cls_df.join(Test_index_df).set_index('Test index')
# cls_df_index.index.name = None 

# Join with the test_index frame 
cls_prediction_df = cls_df.join(xTest)
# Display the MLP classifier
cls_pairplot = sns.pairplot(cls_prediction_df, hue = 'Mlp prediction')


# Obtain training data
moxeeData = sio.loadmat('moxeetrainingdata.mat')    # Load variables from the Moxee dataset
trainingInputs = moxeeData['pressureData']          # Pressure values and differences for every hour in a year
trainingTargets = moxeeData['dataEstimate']         # Estimate of incoming solar energy based on observed data

# Preprocess the training inputs and targets
iScaler = preprocessing.StandardScaler()    # Scaler that removes the mean and scales to unit variance
scaledTrainingInputs = iScaler.fit_transform(trainingInputs)   # Fit and scale the training inputs

tScaler = preprocessing.StandardScaler()
scaledTrainingTargets = tScaler.fit_transform(trainingTargets)

# Create the multilayer perceptron.
# This is where you will be modifying the regressor to try to beat the decision tree
mlp = MLPRegressor(
    hidden_layer_sizes = (1,),     # One hidden layer with 50 neurons
    activation = 'logistic',        # Logistic sigmoid activation function
    solver = 'sgd',                 # Gradient descent
    learning_rate_init = 0.01 ,# Initial learning rate
    )
# 
############################################################### Create the decision tree:
dt_reg = DecisionTreeRegressor(criterion='mse', max_depth = 10) 
dt_reg.fit(scaledTrainingInputs, scaledTrainingTargets)


### MODIFY THE VALUE BELOW ###
noIterations = 98  # Number of iterations (epochs) for which the MLP trains
### MODIFY THE VALUE ABOVE ###

trainingError = np.zeros(noIterations)  # Initialize array to hold training error values

# Train the MLP for the specified number of iterations
for i in range(noIterations):
    mlp.partial_fit(scaledTrainingInputs, np.ravel(scaledTrainingTargets))  # Partial fit is used to obtain the output values after each epoch
    currentOutputs = mlp.predict(scaledTrainingInputs)  # Obtain the outputs for the current MLP using the training inputs
    trainingError[i] = np.sum((scaledTrainingTargets.T - currentOutputs) ** 2) / 2  # Keep track of the error throughout the number of epochs

# Plot the error curve
plt.figure(figsize=(10,6))
ErrorHandle ,= plt.plot(range(noIterations), trainingError, label = 'Error 50HU',  linestyle = 'dotted')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training Error of the MLP for Every Epoch')
plt.legend(handles = [ErrorHandle])
plt.show()

# Obtain test data
testdataset = sio.loadmat('moxeetestdata.mat')
testInputs = testdataset['testInputs']
testTargets = testdataset['testTargets']
scaledTestInputs = iScaler.transform(testInputs)  # Scale the test inputs

# Predict incoming solar energy from the training data and the test cases
scaledTrainingOutputs = mlp.predict(scaledTrainingInputs)
scaledTestOutputs = mlp.predict(scaledTestInputs)

#################################################################### Predict using the bad guy: 
scaledTreeTrainingOutputs = dt_reg.predict(scaledTrainingInputs)
scaledTreeTestOutputs = dt_reg.predict(scaledTestInputs)

# Transform the outputs back to the original values
trainingOutputs = tScaler.inverse_transform(scaledTrainingOutputs)
testOutputs = tScaler.inverse_transform(scaledTestOutputs)
## DT outputs 
treeTrainingOutputs = tScaler.inverse_transform(scaledTreeTrainingOutputs) # -- transform the tree back to real data 
treeTestingOutputs = tScaler.inverse_transform(scaledTreeTestOutputs)

# Calculate and display training and test root mean square error (RMSE)
trainingRMSE = np.sqrt(np.sum((trainingOutputs - trainingTargets[:, 0]) ** 2) / len(trainingOutputs)) / 1000000  # Divide by 1e6 for MJ/m^2
testRMSE = np.sqrt(np.sum((testOutputs - testTargets[:, 0]) ** 2) / len(testOutputs)) / 1000000

## need to add this for the decision tree 
trainingTreeRMSE = np.sqrt(np.sum((treeTrainingOutputs - trainingTargets[:, 0]) ** 2) / len(trainingOutputs)) / 1000000
testTreeRMSE = np.sqrt(np.sum((treeTestingOutputs - testTargets[:, 0]) ** 2) / len(testOutputs)) / 1000000

print("Training RMSE:", trainingRMSE, "MJ/m^2")
print("Test RMSE:", testRMSE, "MJ/m^2")
##################################################################### Print the tree RMSE:
print("Decision Tree training RMSE:", trainingTreeRMSE, 'MJ/m^2')
print("Decision Tree Test RMSE:", testTreeRMSE, 'MJ/m^2')
day = np.array(range(1, len(testTargets) + 1))

# Plot training targets vs. training outputs
plt.figure(figsize=(10,6))
trainingTargetHandle ,= plt.plot(day, trainingTargets / 1000000, label = 'Target values')
trainingOutputHandle ,= plt.plot(day, trainingOutputs / 1000000, label = 'Outputs 50HU',  linestyle = 'dotted')
plt.xlabel('Day')
plt.ylabel(r'Incoming Solar Energy [$MJ / m^2$]')
plt.title('Comparison of MLP Training Targets and Outputs')
plt.legend(handles = [trainingTargetHandle, trainingOutputHandle])
plt.show()

# Plot test targets vs. test outputs -- student 
plt.figure(figsize=(10,6))
testTargetHandle ,= plt.plot(day, testTargets / 1000000, label = 'Target values')
testOutputHandle ,= plt.plot(day, testOutputs / 1000000, label = 'Outputs 50HU',  linestyle = 'dotted')
plt.xlabel('Day')
plt.ylabel(r'Incoming Solar Energy [$MJ / m^2$]')
plt.title('Comparison of MLP Test Targets and Outputs')
plt.legend(handles = [testTargetHandle, testOutputHandle])
plt.show()

###################################################################### Plot the tree regressor vs. test outputs
plt.figure(figsize=(10,6))
testTreeTargetHandle, = plt.plot(day, testTargets / 1000000, label = 'Target values')
testTreeOutputHandle, = plt.plot(day, treeTestingOutputs / 1000000, label = 'Decision tree', linestyle = 'dotted')
plt.xlabel('Day')
plt.ylabel(r'Incoming Solar Energy [$MJ / m^2$]')
plt.title('Comparison of Decision Tree Test Targets and Outputs')
plt.legend(handles = [testTreeTargetHandle, testTreeOutputHandle])
plt.show()


#INITIALIZE 
from sklearn.svm import LinearSVR
svm_clf = LinearSVR(C=0.6, loss='squared_epsilon_insensitive')
svm_clf.fit(scaledTrainingInputs, np.ravel(scaledTrainingTargets)) 

# PREDICT the training outputs and the test outputs
scaledTrainingOutputs = svm_clf.predict(scaledTrainingInputs)
scaledTestOutputs = svm_clf.predict(scaledTestInputs)


trainingOutputs = tScaler.inverse_transform(scaledTrainingOutputs)
testOutputs = tScaler.inverse_transform(scaledTestOutputs)

 #Calculate and display training and test root mean square error (RMSE)
trainingsvmRMSE = np.sqrt(np.sum((trainingOutputs - trainingTargets[:, 0]) ** 2) / len(trainingOutputs)) / 1000000  # Divide by 1e6 for MJ/m^2
testsvmRMSE = np.sqrt(np.sum((testOutputs - testTargets[:, 0]) ** 2) / len(testOutputs)) / 1000000

#### PLOTTING
plt.rcParams["figure.figsize"] = (10,6)
day = np.array(range(1, len(testTargets) + 1))

testTargetHandle, = plt.plot(day, testTargets / 1000000, label = 'Target Values')
testsvmOutputHandle, = plt.plot(day, testOutputs / 1000000, label = 'SVM Prediction', linestyle = 'dotted')
plt.xlabel('Day')
plt.ylabel(r'Incoming Solar Energy [$MJ / m^2$]')
plt.title('Comparison of Prediction Targets and SVM Predictions')
plt.legend(handles = [testTargetHandle, testsvmOutputHandle])
plt.show()

print("Support Vector Machine RMSE values and Plots")
print("Training RMSE:", trainingsvmRMSE, "MJ/m^2")
print("Test RMSE:", testsvmRMSE, "MJ/m^2")

# Modify this neural network 
mlp = MLPRegressor(
    hidden_layer_sizes = (1,),     # One hidden layer with 50 neurons
    activation = 'logistic',        # Logistic sigmoid activation function
    solver = 'sgd',                 # Gradient descent
    learning_rate_init = 0.01 ,# Initial learning rate
    )
# 
############################################################### Create the decision tree:
dt_reg = DecisionTreeRegressor(criterion='mse', max_depth = 10) 
dt_reg.fit(scaledTrainingInputs, scaledTrainingTargets)


### MODIFY THE VALUE BELOW ###
noIterations = 98  # Number of iterations (epochs) for which the MLP trains
### MODIFY THE VALUE ABOVE ###

trainingError = np.zeros(noIterations)  # Initialize array to hold training error values

# Train the MLP for the specified number of iterations
for i in range(noIterations):
    mlp.partial_fit(scaledTrainingInputs, np.ravel(scaledTrainingTargets))  # Partial fit is used to obtain the output values after each epoch
    currentOutputs = mlp.predict(scaledTrainingInputs)  # Obtain the outputs for the current MLP using the training inputs
    trainingError[i] = np.sum((scaledTrainingTargets.T - currentOutputs) ** 2) / 2  # Keep track of the error throughout the number of epochs
    
# Predict 
scaledTrainingOutputs = mlp.predict(scaledTrainingInputs)
scaledTestOutputs = mlp.predict(scaledTestInputs)
#Training output conversion    
trainingOutputs = tScaler.inverse_transform(scaledTrainingOutputs)
testOutputs = tScaler.inverse_transform(scaledTestOutputs)

#RMSE calculation 
trainingRMSE = np.sqrt(np.sum((trainingOutputs - trainingTargets[:, 0]) ** 2) / len(trainingOutputs)) / 1000000  # Divide by 1e6 for MJ/m^2
testRMSE = np.sqrt(np.sum((testOutputs - testTargets[:, 0]) ** 2) / len(testOutputs)) / 1000000
    
# Plot the error curve
plt.figure(figsize=(10,6))
ErrorHandle ,= plt.plot(range(noIterations), trainingError, label = 'Error 50HU',  linestyle = 'dotted')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training Error of the MLP for Every Epoch')
plt.legend(handles = [ErrorHandle])
plt.show()

print("MLP Training and test RMSE values:")
print("Training RMSE: " , trainingRMSE)
print("Test RMSE: " , testRMSE)