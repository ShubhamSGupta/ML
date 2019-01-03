# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
filename = 'data/breastCancer.data.csv'
names = ['Id', ' Clump Thickness', 'cell size', 'cell shape', 'Marginal Adhesion', 'Single Epithelial Cell Size ',
         'Bare_Nuclei', 'Bland Chromatin ', 'Normal Nucleoli', 'Mitoses', 'Class']
dataframe = read_csv(filename, names=names)
dataframe = dataframe[dataframe.Bare_Nuclei != '?']
array = dataframe.values
X = array[:,1:10]
Y = array[:,10]
Y = Y.astype('int')
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
