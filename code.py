# --------------
# Code starts here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew

#### Data 1
# Load the data
data1 = pd.read_csv(path)
# Overview of the data
print(data1.info())
print(data1.describe())
# Histogram showing distribution of car prices
data1.hist(column='price', figsize=(10,10))
# Countplot of the make column
ax = sns.countplot(x="make", data=data1)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
# Jointplot showing relationship between 'horsepower' and 'price' of the car
ax1 = sns.jointplot(x="horsepower", y="price", data=data1, kind="reg")
# Correlation heat map
ax2 = sns.heatmap(data1.corr())
# boxplot that shows the variability of each 'body-style' with respect to the 'price'
ax3 = sns.boxplot(x="body-style", y="price", data=data1)

#### Data 2

# Load the data
data2 = pd.read_csv(path2)
data2 = data2.replace("?", "NaN")
# Impute missing values with mean
mean_imputer = Imputer(missing_values="NaN", strategy='mean')
data2[['normalized-losses']] = mean_imputer.fit_transform(data2[['normalized-losses']])
data2[['horsepower']] = mean_imputer.fit_transform(data2[['horsepower']])
# Skewness of numeric features
num_col = data2._get_numeric_data().columns
for i in num_col :
    if skew(data2[i]) > 1 :
        data2[i] = np.sqrt(data2[i])

# Label encode 
label_encoder = LabelEncoder()
cat_col = data2.select_dtypes(include='object').columns
for i in cat_col :
    data2[i] = label_encoder.fit_transform(data2[i])

#New Col
data2['area'] = data2['height'] * data2['width']
print(data2['area'])

# Code ends here


