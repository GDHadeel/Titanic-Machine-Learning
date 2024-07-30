# Titanic-Machine-Learning

## Task 7: Use machine learning to create a model that predicts which passengers survived the Titanic or not Using Logistic Regression to train the model, and accuracy score to evaluate the model

## Description
This project aims to introduce the most important steps of data analysis and explore the different stages. We will use the data of Titanic survivors available on the Kaggle website.

## Dataset
The dataset is available on Kaggle:
[Download the dataset](https://www.kaggle.com/competitions/titanic/data)

## Project Structure
1. Data Loading and Inspection
2. Data Preprocessing
3. Data Analysis
4. Model Building
5. Model Evaluation

## Dependencies
To complete this project, you need the following Python libraries:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

## Google Colab Notebook
Click the link below to open the notebook in Google Colab and run the code.
[Google Colab](https://colab.research.google.com/drive/1FV-1R1Un7y-8ntnmimNX4hH_kHkoQVk-#scrollTo=EGhTRZYwYKm4)

## Code 
### 1. Data Loading and Inspection
We will load the data and inspect it to ensure it has been read correctly.
```python
import pandas as pd

# Load the dataset into a Pandas DataFrame
data = pd.read_csv('/content/train.csv')

# Display the first 5 rows of the dataset
data.head()

# Show the number of rows and columns
data.shape
```
     
### 2. Data Preprocessing
Data preprocessing involves handling missing values, dropping unnecessary columns, and encoding categorical variables.
##### 1. Handling Missing Data
* Check for missing values:
```python
data.isnull().sum()
```
* Fill missing values for the Age column with the mean:
```python
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Age'].isnull().sum()
```
* Drop the Cabin column due to a large number of missing values:
```python
data = data.drop(['Cabin'], axis=1)
data.head()
```
* Fill missing values in the Embarked column with the mode:
```python
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Embarked'].isnull().sum()
```
##### 2. Drop Unnecessary Columns
* Drop columns that are not useful for prediction:
```python
data = data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
data.head()
```
##### 3. Encode Categorical Columns
* Convert categorical text values to numerical values:
```python
data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
data.head()

```
##### 4. Handle Duplicates
* Check for and remove any duplicate rows:
```python
# Check for duplicates
data.duplicated().sum()

# Drop duplicate rows
data.drop_duplicates(inplace=True)
```

### 3. Data Analysis
* Perform exploratory data analysis to understand the dataset and relationships between features.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Summary statistics for numerical columns
data.describe()

# Correlation matrix
data.corr()['Survived']

# Count plot for 'Survived'
sns.countplot(x='Survived', data=data)
plt.show()

# Count plot for 'Sex'
sns.countplot(x='Sex', data=data)
plt.show()

# Count plot for survival based on gender
sns.countplot(x='Sex', hue='Survived', data=data)
plt.show()

# Count plot for survival based on Pclass
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.show()
```
    
### 4. Model Building
Prepare the data for machine learning by separating features and target variables, then train a logistic regression model.
##### 1. Separate Features and Target
```python
x = data.drop(columns=['Survived'])
y = data['Survived']
```
##### 2. Split Data into Training and Testing Sets
```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```
##### 3. Train the Model
```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

### 5. Model Evaluation
Evaluate the model's performance using accuracy score.
```python
from sklearn.metrics import accuracy_score

# Predict on the test data
y_pred = model.predict(x_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

## Acknowledgements
https://www.datacamp.com/blog/classification-machine-learning


