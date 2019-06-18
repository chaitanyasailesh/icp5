import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
# Only uses the passed col[string list] to make data frame
# loads sheets from 63 to 81 as a dataframe by using usecols
dataframe = pd.read_csv('train.csv', sep=',', usecols=(62, 80))


y = dataframe['SalePrice']
x = dataframe['GarageArea']
# Return a tuple representing the dimensionality of the DataFrame
print('original shape of datashape', dataframe.shape)
# plotting points
plt.scatter(x, y)
plt.title("original dataframe")
plt.ylabel("SalesPrice")
plt.xlabel("GarageArea")
# displays the diagram
plt.show()

# computes the relative Z-score of the input data, relative to the sample mean and standard deviation
z = np.abs(stats.zscore(dataframe))
# A point beyond which there is a change in the manner a program executes
threshold = 3
print(np.where(z > 3))
modified_df = dataframe[(z < 3).all(axis=1)]

y = modified_df['SalePrice']
x = modified_df['GarageArea']
print('after removing outliers', modified_df.shape)

plt.scatter(x, y)
plt.title("after deleting outliers")
plt.ylabel("SalesPrice")
plt.xlabel("GarageArea")
plt.show()