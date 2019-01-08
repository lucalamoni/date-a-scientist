import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Create your df here:

############ Import dataset
df = pd.read_csv("profiles.csv") 

############ Explore data
print(df.income.value_counts())

############Subset data based on main ethnicity categories
subset = df.query('ethnicity == ["white","asian","hispanic / latin","black","other","hispanic / latin, white","indian"]')

#Subset further to exclude incomes of -1 from the analysis
subset2 = subset[subset.income > 0]


#############Figure 1 Distribution of main ethnicity of users
counts = [6522,914,554,483,359,266,189]
labels = ['white','asian','hispanic / latin','black','other','hispanic / latin, white','indian']
plt.bar(labels, counts)
plt.xlabel("Ethnicity")
plt.ylabel("Number of Users")
plt.show()

##############Figure 2 Distribution of user sexes
counts2 = [6778,2509]
labels2 = ['Males','Females']
plt.bar(labels2, counts2)
plt.xlabel("Sex")
plt.ylabel("Number of Users")
plt.show()

############### Ethnicity and sex Mapping
ethnicity_mapping = {"white": 0, "asian": 1, "hispanic / latin": 2, "black": 3, "other": 4, "hispanic / latin, white": 5, "indian":6}
sex_mapping = {"m":0,"f":1}
subset2["ethnicity_code"] = subset2.ethnicity.map(ethnicity_mapping)
subset2["sex_code"] = subset2.sex.map(sex_mapping)

############## Dividing train and test data

x = subset2[['ethnicity_code','income']]
y = subset2[['sex_code']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

############## Multiple Linear Regression 
mlr = LinearRegression()
mlr.fit(x_train, y_train) 
y_predict = mlr.predict(x_test)

# Visualisation
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("real sex")
plt.ylabel("predicted sex")
plt.title("real vc predicted income")
plt.show()

#Test Score
print("Test score:")
print(mlr.score(x_test, y_test))
#Test score:
#-0.0006916384214745719


############## K nearest neighbour regression
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 20, weights = "distance")
regressor.fit(x_train,y_train)

print("K Test score:")
print(regressor.score(x_test, y_test))
#K Test score:
#-0.16298310826828666


############# K nearest neighbour Classification
from sklearn.neighbors import KNeighborsClassifier

# Visualisation of k with highest accuracy
accuracy = [None]*199
for i in range(1,200):
	classifier = KNeighborsClassifier(n_neighbors = i)
	classifier.fit(x_train, y_train)
	accuracy[i]= classifier.score(x_test,y_test)

plt.plot(range(1,200),accuracy)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.show()


classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)

print(accuracy_score(y_test,y_predict))
print(recall_score(y_test,y_predict))
print(precision_score(y_test,y_predict))
print(f1_score(y_test,y_predict))
#0.7238966630785791
#0.003929273084479371
#0.25
#0.007736943907156672

############ Support Vector Machines
from sklearn.svm import SVC

classifierSVC = SVC(kernel = 'rbf', gamma = 0.9, C = 2)
classifierSVC.fit(x_train, y_train)
y_predictSVC = classifierSVC.predict(x_test)

print(accuracy_score(y_test,y_predictSVC))
print(recall_score(y_test,y_predictSVC))
print(precision_score(y_test,y_predictSVC))
print(f1_score(y_test,y_predictSVC))
#0.7255113024757804
#0.0
#0.0
#0.0
