%matplotlib inline
import numpy as np
import pylab
from scipy.stats import norm
import matplotlib.mlab as mlab
import math
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from operator import itemgetter
import pandas as pd
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 
from matplotlib.ticker import AutoMinorLocator
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
from IPython.display import display
#import importlib
#importlib.reload(norm)
import re
from matplotlib import rcParams
import pandas as pd
rcParams.update({'figure.autolayout': True})

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def neurosesin_ccs_features(filename1, filename2, title, graphname):
    allfeatures = []
    
    datadict = {}
    #path = "/Users/pxq/Documents/data/li/npy/ccs_mini/"
    path = "/Users/pxq/Documents/data/li/daacp_alex/md/ccs/dihes/neurosesin/"
    featurelist = ["phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4", "chi5"]
    columnname = []
    filenames = [filename1, filename2]
    allfeatures = pd.DataFrame()
    for count, filename in enumerate(filenames):
        features = []
        for listname in featurelist: 
            filenamefull = path + filename + listname + ".list"
            x = []
            with open(filenamefull, 'r') as f:
                for line in f:
                    line = line.replace('acth_homo_d16lys_p5_', ' ')
                    line = line.replace(":", "")
                    line = re.sub("\s\s+", ' ', line)
                    words = line.split()
                    x.append(words)   ## read the whole data set in x
            x = np.array(x)
            columnname = [(listname + str(i)) for i in range(0, len(x[0]))]
            dropcolumns = [(listname + str(2*i)) for i in range(0, len(x[0])/2) ]
            inputs = pd.DataFrame(x, columns= columnname)
            inputs.drop(dropcolumns, axis=1, inplace = True)
            newcolumns = [(listname + str(i)) for i in range(1, inputs.shape[1]+1)]
            #print(len(inputs.values), inputs.values.size/len(inputs.values), len(newcolumns), \
            #      newcolumns.size/len(newcolumns))
            inputs = pd.DataFrame(inputs.values, columns=newcolumns)
            
            features = inputs if(len(features)== 0) else pd.concat([features, inputs], axis=1)
        labels = pd.DataFrame(np.full(len(features), count), columns= ["dl_label"])
        sequence = pd.DataFrame([i for i in range(1, len(features)+1)], columns= ["pdb_id"])
        features = pd.concat([features, labels, sequence], axis=1)
        allfeatures = features if(len(allfeatures)== 0) else pd.concat([allfeatures, features], axis=0)
       
    #print(len(allfeatures))
    #display(allfeatures.head(5))
    #display(allfeatures.tail(5))
    ccs_regression(allfeatures)
    #return(allfeatures)

def feature_selection():
    pass

def ccs_regression(data):
    inner_cv = KFold(n_splits=2, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=2, shuffle=True, random_state=1)
    reg = linear_model.Lasso(alpha = 0.1)
    
    alpha = [0.01, 0.1, 0.5]
    tol = [0.0001, 0.001]
    params = [
        {
            'regressor__alpha': alpha,
            'regressor__tol': tol
        },
    ]
    pipeline = Pipeline([('featuresel', RFECV(estimator=reg, scoring='r2')), ('regressor', reg) ])
    grid_search = GridSearchCV(pipeline, params, cv=inner_cv)
    nested_score = cross_val_score(grid_search, X=data[data.columns[1:data.shape[1]]],  y=data[data.columns[0]], cv=outer_cv).mean()


neurosesin_ccs_features("neurotensin_p3_d11trp_", "neurotensin_p3_lform_", "neurotensin_p3_phi", "neurotensin_p3_phi")

