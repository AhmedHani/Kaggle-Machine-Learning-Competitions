__author__ = 'Ahmed Hani Ibrahim'

import pandas as pnd
import seaborn as sb
import matplotlib.pyplot as plt

sb.set()
data = pnd.read_csv("./train.csv", parse_dates=['Dates'], header=0)
data['Year'] = data['Dates'].map(lambda x: x.year)
data['Week'] = data['Dates'].map(lambda x: x.week)
data['Hour'] = data['Dates'].map(lambda x: x.hour)

data = pnd.crosstab(data.Category, data['Year'])

sb.heatmap(data, square=True, linewidths=1, cmap='YlGnBu')

plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
