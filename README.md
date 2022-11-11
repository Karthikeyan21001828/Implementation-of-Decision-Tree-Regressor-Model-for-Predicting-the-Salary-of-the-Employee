# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries from python.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply to the model from the dataset.
5. Predict the values of the arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model from the dataset.
7. Predict the values of array
8. Apply it to the new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Karthikeyan.K
RegisterNumber: 212221230046
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![op1](https://user-images.githubusercontent.com/93427303/198190086-06a9a267-5001-4c50-bff3-795d32668d97.png)

![op2](https://user-images.githubusercontent.com/93427303/198190087-5ec4c936-1135-4a27-a086-ed04902ca198.png)

![op3](https://user-images.githubusercontent.com/93427303/198190095-025daa40-cddd-4dfe-9d4e-9a11630fc556.png)

![op4](https://user-images.githubusercontent.com/93427303/198190112-806ef154-8292-4890-b539-ae19fb568b64.png)

![op5](https://user-images.githubusercontent.com/93427303/198190134-736d2b46-641f-4be4-acbd-2bddc1e101da.png)

![op6](https://user-images.githubusercontent.com/93427303/198190156-f517a4b4-abfc-4ea9-884e-13d5d1644b55.png)

![op7](https://user-images.githubusercontent.com/93427303/198190175-b256d9fc-a2c5-49be-ab2d-9033512bb693.png)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
