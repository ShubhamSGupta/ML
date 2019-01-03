from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

filename = 'data/breastCancer.data.csv'
names = ['Id', ' Clump Thickness', 'cell size', 'cell shape', 'Marginal Adhesion', 'Single Epithelial Cell Size ',
         'Bare_Nuclei', 'Bland Chromatin ', 'Normal Nucleoli', 'Mitoses', 'Class']
dataframe = read_csv(filename, names=names)
dataframe = dataframe[dataframe.Bare_Nuclei != '?']
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
Y = Y.reshape(-1,1)
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
binarizer = Binarizer(threshold=3.0).fit(Y)
binaryY = binarizer.transform(Y)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])
print(binaryY[0:5,:])