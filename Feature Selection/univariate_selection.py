# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest,chi2

filename = 'data/breastCancer.data.csv'
names = ['Id', ' Clump Thickness', 'cell size', 'cell shape', 'Marginal Adhesion', 'Single Epithelial Cell Size ',
         'Bare_Nuclei', 'Bland Chromatin ', 'Normal Nucleoli', 'Mitoses', 'Class']
dataframe = read_csv(filename, names=names)
dataframe = dataframe[dataframe.Bare_Nuclei != '?']
array = dataframe.values
# separate array into input and output components
X = array[:,1:10]
Y = array[:,10]
Y = Y.astype('int')
test = SelectKBest(score_func=chi2, k=9)
fit = test.fit(X, Y)
# summarize transformed data
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features[0:5,:])
