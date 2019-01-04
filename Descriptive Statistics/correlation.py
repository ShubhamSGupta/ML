from matplotlib import pyplot
from pandas import read_csv
import numpy

filename = 'data/breastCancer.data.csv'
names = ['Id', ' Clump Thickness', 'cell size', 'cell shape', 'Marginal Adhesion', 'Single Epithelial Cell Size ',
         'Bare Nuclei', 'Bland Chromatin ', 'Normal Nucleoli', 'Mitoses', 'Class']

data = read_csv(filename, names=names)
correlation = data.corr()
#plotting corelation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmax=1, vmin=-1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()