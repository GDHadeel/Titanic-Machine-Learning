# Titanic-Machine-Learning

## Task 7: use machine learning to create a model that predicts which passengers survived the Titanic or not Using Logistic Regression to train the model, and accuracy score to evaluate the model

## Description
This project aims to introduce the most important steps of data analysis and explore the different stages. We will use the data of Titanic survivors available on the Kaggle website.

### Dataset
The dataset is available on Kaggle:
[Download the dataset](https://www.kaggle.com/competitions/titanic/data)

### Dependencies
To complete this project, you need the following Python libraries:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

### Steps
#### 1. Importing the Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
     
#### 2. Reading the Data
First, load the data from the CSV file into a Pandas DataFrame:

```python
data = pd.read_csv('/content/train.csv')

from google.colab import drive
drive.mount('/content/drive')

data.head()
```

#### 3. Data Preprocessing
##### 1. Dealing with Missing Data
* Identify missing values:

```python
data.info()
data.isnull().sum()
```
    
* Fill missing values in the Age column with the median:

```python
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Age'].isnull().sum()
```
    
* Drop the Cabin column due to many missing values:

```python
data = data.drop(['Cabin'], axis=1)
data.head()
```

* Fill missing values in the Embarked column with the mode:

```python
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Embarked'].isnull().sum()
```

##### 2. Drop Useless Columns
Drop the PassengerId and Name columns:

```python
data = data.drop(columns=['PassengerId', 'Name'])
data.head()
```

##### 3. Encode Categorical Columns
Convert categorical columns to numerical values:

```python
data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
data.head()
```

##### 4. Dealing with Duplicates
Check and drop duplicates:

```python
duplicates = data.duplicated().sum()
print(f'Number of duplicate rows: {duplicates}')
data.drop_duplicates(inplace=True)
```

#### 4. Data Analysis
Explore the data with statistical analysis and visualization:

```python
data.describe()
non_numeric_columns = data.select_dtypes(exclude=[float, int]).columns
print(f'Non-numeric columns: {non_numeric_columns}')
data['Ticket'] = pd.to_numeric(data['Ticket'], errors='coerce')  
print(data.corr()['Survived'])
sns.countplot(x='Survived', data=data)
sns.countplot(x='Sex', data=data)
sns.countplot(x='Sex', hue='Survived', data=data)
sns.countplot(x='Pclass', hue='Survived', data=data)
```

#### 5. Model Building
##### Separating Features and Target
Separate features and target variable:

```python
X_train = data.drop(columns=['Survived'], axis=1)
y_train = data['Survived']
```
    
#### 6. Model Training
Train a Logistic Regression model:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
```

#### 7. Model Evaluation
Evaluate the model with the accuracy score:

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

## Acknowledgements
https://www.youtube.com/watch?v=1tNER04Ytyc&list=LL&index=8&t=874s

https://www.youtube.com/watch?v=kAWNSolkkqg

