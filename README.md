## Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

### ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file
## CODE
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()
sns.heatmap(data.isnull(),cbar=False)
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
data['Embarked']=data['Embarked'].fillna('S')
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
data.head(11)
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sns.heatmap(data.isnull(),cbar=False)
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5, criterion='entropy')

my_forest.fit(X_train, y_train)

target_predict = my_forest.predict(X_test)
accuracy = accuracy_score(y_test, target_predict)
mse = mean_squared_error(y_test, target_predict)
r2 = r2_score(y_test, target_predict)

print("Random forest accuracy: ", accuracy)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2) Score: ", r2)
```
## OUTPUT:
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/a9160843-6f3a-47c1-9e39-2c395e5fdaca)
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/6093da9f-84ce-4b82-8b55-681ea8770784)
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/56085ba9-9077-40c6-a775-e90995c82e92)
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/e5da37d2-722e-4087-9b15-9887eedd3f94)
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/987dddf2-6e53-4a2c-92d3-dd8bbf972eaa)
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/fa98b106-3d7a-4df4-b1e4-87d9cf9cbac0)
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/bf518414-482f-4e88-869f-7fd9bad104f2)
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/d2ed4509-8095-435c-9489-a305d30abcd9)
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex-07/assets/131433133/e32d9e8d-f617-4719-8d8e-6bcabd4ab3b8)
## RESULT:
Thus, Successfully performed the various feature selection techniques on a given dataset.
