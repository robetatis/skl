# load packages

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

# combine into single dataframe
df = rain
df['atmpress'] = press['pressure']
df['airtemp'] = temp['temperature']
df['relhum'] = temp['relative_humidity']

# remove no data (labeled as -999) and na's
df = df[df != -999]
df = df.dropna()

# plot hourly variables
plt.close()
plt.figure()
plt.subplots_adjust(top=0.9, bottom=0.9)
plt.subplot(311)
plt.plot(df['rain'], 'b.', markersize=3)
plt.ylabel('Rain [mm/hour]')
plt.yticks(np.linspace(0, 40, 5))
plt.subplot(312)
plt.plot(df['airtemp'], '-r', linewidth=0.3)
plt.yticks(np.linspace(-10, 40, 6))
plt.ylabel('Temp. [ºC]')
plt.subplot(313)
plt.plot(df['atmpress'], '-g', linewidth=0.3)
plt.ylabel('Atm. Press [hPa]')
plt.yticks(np.linspace(980, 1040, 5))
plt.suptitle('Hourly data')
plt.show()

# aggregate to daily values using different statistics (max, median, mean)
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
# ...

# perceptron learning rule: w = w + alpha(y - hw(x))*x,
# with hw(x) = w0 + w1*x2 + w2*x2 >= 0 -> y = 1
#                                 < 0  -> y = 0


model = Perceptron()
#model = KNeighborsClassifier(n_neighbors=1)

data = []
for index, row in df.iterrows():
    data.append({
        "evidence": [float(ij) for ij in row[1:4]],
        "label": 1 if row[0] > 0 else 0
        })


# split training and testing sets
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# train model
X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
model.fit(X_training, y_training)

# predict on testing set
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)

# prediction quality
corr = 0
incorr = 0
tot = 0
for actual, predicted in zip(y_testing, predictions):
    tot += 1
    if actual == predicted:
        corr += 1
    else:
        incorr += 1

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {corr}")
print(f"Incorrect: {incorr}")
print(f"Accuracy: {100 * corr / tot:.2f}%")
