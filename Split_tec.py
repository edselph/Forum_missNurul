import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"S:/Binus/Fundamentals of Data Science/task/Forum/venv/scr/toy_dataset.csv")

# print(df.head())

columns = ['City', 'Gender', 'Age', 'Income']

target_columns =['Illness']

x = df[columns]
y = df[target_columns]

x = pd.get_dummies(x)

"""Split"""

x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=1 , test_size=0.2)

model = DecisionTreeClassifier(random_state=1, criterion='entropy')
model.fit(x_train, y_train)

y_predict = model.predict(x_valid)
