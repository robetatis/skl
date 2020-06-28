import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier


# import data
dateparser = lambda x: pd.to_datetime(x, format='%Y%m%d%H')
press = pd.read_csv('press.txt', delimiter=';', usecols=[1, 4], names=['dateTime', 'pressure'], 
                    skiprows=1, parse_dates=['dateTime'], date_parser=dateparser, index_col='dateTime')
temp = pd.read_csv('temp.txt', delimiter=';', usecols=[1, 4, 5], names=['dateTime', 'temperature', 'relative_humidity'], 
                    skiprows=1, parse_dates=['dateTime'], date_parser=dateparser, index_col='dateTime')
rain = pd.read_csv('rain.txt', delimiter=';', usecols=[1, 4], names=['dateTime', 'rain'], 
                    skiprows=1, parse_dates=['dateTime'], date_parser=dateparser, index_col='dateTime')

df = rain
df['atmpress'] = press['pressure']
df['airtemp'] = temp['temperature']
df['relhum'] = temp['relative_humidity']

# clean data
df = df[df != -999]


plt.plot(df)
plt.show()


# plot hourly variables


# aggregate to daily values using different statistics (max, median, mean)



# build classification model

# perceptron learning rule: w = w + alpha(y - hw(x))*x,
# with hw(x) = w0 + w1*x2 + w2*x2 >= 0 -> y = 1
#                                 < 0  -> y = 0


model = Perceptron()
#model = KNeighborsClassifier(n_neighbors=1)

with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })


# Separate data into training and testing groups
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# Train model on training set
X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
model.fit(X_training, y_training)

# Make predictions on the testing set
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)

# Compute how well we performed
correct = 0
incorrect = 0
total = 0
for actual, predicted in zip(y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
