# Feature Extraction with PCA
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
filename = 'data/breastCancer.data.csv'
names = ['Id', ' Clump Thickness', 'cell size', 'cell shape', 'Marginal Adhesion', 'Single Epithelial Cell Size ',
         'Bare_Nuclei', 'Bland Chromatin ', 'Normal Nucleoli', 'Mitoses', 'Class']
dataframe = read_csv(filename, names=names)
dataframe = dataframe[dataframe.Bare_Nuclei != '?']
array = dataframe.values
X = array[:,1:10]
Y = array[:,10]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: ",fit.explained_variance_ratio_)
print(fit.components_)