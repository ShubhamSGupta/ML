# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
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
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
#print(fit)
print("Num Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_)
