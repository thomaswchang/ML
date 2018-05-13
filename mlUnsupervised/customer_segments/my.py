# ------------------------------
# Import libraries
# ------------------------------
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# from IPython.core.debugger import Tracer
# To run through a debugger, add: Tracer()()  before the line you want to stop. c, n. To see output, print VAR
 
    

    
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report, f1_score, mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time
from sklearn import tree
#import graphviz 


class MyDataExplorationUtil:
    """
    Study each features and its characteristics: Name, type, % missing values, noise and type of noise, type of distribution (skewed or normal), summary statistics

    Visualize the data: Histogram, line chard, scatter, whiseter
    
    Study the correlation between featuers: seborn pair plots & pearson correlation
    """

    # Example: plotDistributionMultipleFields(data, ['capital-gain','capital-loss'])
    def plotDistributionMultipleFields(self, df, fields):
        # Create figure
        fig = plt.figure(figsize = (11,5));

        for i, feature in enumerate(fields):
            ax = fig.add_subplot(1, 2, i+1)
            ax.hist(df[feature], bins = 25, color = '#00A0A0')
            ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
            ax.set_xlabel("Value")
            ax.set_ylabel("Number of Records")
            #ax.set_ylim((0, 2000))
            #ax.set_yticks([0, 500, 1000, 1500, 2000])
            #ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Example: plotDistibution(df, 'Age')
    def plotDistibutionSingleField(self, df, field):
        sns.distplot(df[field])
    
    # Method bins the DF by the values in the column
    # Example: plotCount(data, 'Survived')
    def plotCount(self, df, primaryField):
        sns.countplot(x=primaryField, data=df)
    
    # Method bins the DF by the values of the primary-field and secondary-field
    def plotCountWithPivot(self, df, primaryField, secondaryField):
        sns.countplot(x=primaryField, hue=secondaryField, data=df)
        # An alternative text approach is: df.groupby(['Survived', 'Pclass]).size()
    
    # Example: plotHistogram(df, 'Age', 10)
    def plotHistogram(self, df, field, numBins, xLabel, yLabel):
        df[field].hist(bins=numBins)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
    
    # Example: plotPairs(df, ['MEDV', 'RM', 'LSTAT', 'PTRATIO'])
    def plotPairs(self, df, fields):
        sns.pairplot(df[fields])
    
    # Note: K data loss investigation used this
    def plotScatter(self, df, xAxisName, yAxisName):
        sns.lmplot(x=xAxisName, y=yAxisName, data=df, fit_reg=True)

    # Use this to see the data distributions for all the features.  Shoes the pearson's correlation coefficient.  The "tighter, less variance" the line, the higher the value.
    def plotScatterMatrix(self, data):
        pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
        
    # Good to show number of nulls
    # condition = pd.isnull(df)  //True will result in a tick on the plot
    # plotHeatMap(condition)
    def plotHeatMap(self, condition):
        sns.heatmap(condition, yticklabels=False, cbar=False)
        
    def plotDensitySingleField(self, df, field):
        # Can have multiple kdeplot statements here...
        sns.kdeplot(df[field])
    
    def plotDensityRelationshipBetween2Field(self, df, field1, field2):
        sns.kdeplot(df[field1], df[field2])
    
    def plotViolin(self, df, xField, yField):
        sns.violinplot(x=xfield, y=yField, data=df)
        
    # Method tels how many rows and type of each df column
    def info(self, df):
        df.info()
    
    def stats(self, df):
        print "Dataset has {} data points with {} variables each.".format(*df.shape)
        return df.describe()

    # Method shows the correlation between the columns of the df.
    def correlation(self, df):
        return df.corr()
    
    # Method returns how many rows in a field
    def showNumRows(self, df, field):
        return df[field].sum()
    
    # Method show how many nulls 
    def showNumNulls(self, df):
        pd.isnull(df).sum(axis=0)
        
    # Method returns a set of elements from the df
    # sampleData(df, [0,10,20])
    def sampleData(self, df, indices):
        return pd.DataFrame(df.loc[indices], columns = df.keys()).reset_index(drop = True)

class MyDataCleaningUtil:
    """
    what's the outliers:
        what's the range for outlier?
        remove points outsidfe of outlier
        missing values -> imputation (a) by group (b) by value (c) just drop
    """
    def readCSV(self, fileName):
        return pd.read_csv(fileName)
    
    # Method show how many nulls 
    def showNumNulls(self, df):
        pd.isnull(df).sum(axis=0)
    
    # Example: featuresRaw = myDataCleaningUtil.dropCols(df, 'Survivied')
    def dropCols(self, df, field):
        return df.drop(field, axis=1)
    
    # Example: Drop rows from df based on whether condition returns true or false
    #   dropRowsBasedOnCondition(titanicDataDFDropped, (np.isfinite(titanicDataDFDropped['Age'])) & (True))
    def dropRowsBasedOnCondition(self, df, condition):
        return df[condition]
    
    # Example: Select a field from the df
    #   yLabel = myDataCleaningUtil.selectCol(df, 'Survived')
    def selectCol(self, df, field):
        return df[field]

    # Method find data points that are considered to be outliers based on the Tukey method
    # Reference: http://colingorrie.github.io/outlier-detection.html
    # Used: http://localhost:8888/notebooks/projects/machine-learning/projects/customer_segments/customer_segments.ipynb
    def removeOutlierDataPoints(self, df, stepMultiplier=1.5):
        outlierDataIndex  = []
        outlierDataAll = []
        for feature in df.keys():
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            iqr = Q3 - Q1    
            step = stepMultiplier*iqr
            print "Data points considered outliers for the feature '{}':".format(feature)
            outlierData = df[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step))]    
            #display(outlierData)
            outlierDataAll.append(outlierData)
            outlierDataIndex = outlierDataIndex + outlierData.index.values.tolist()
    
        return (df.drop(df.index[outlierDataIndex]).reset_index(drop = True), outlierDataAll)


    # Side: DF vs panda.Series vs numpy.ndarray 
    # data : DataFrame = pd.readCSV()
    # cols: Panda.Series = data[field]
    # output_horizonal_matrix: numpy.ndarray = data[field].values
    # output_vertical_matrix: numpy.ndarray = data[field].values.reshape(-1,1)
    # X_train: numpy.ndarray ,,,, = train_test_split(data)
    # lreg.fit(train_set['sqft_living'].values.reshape(-1,1), train_set['price'])


class MyFeatureTransformUtil:
    """
    logTransform
    minMaxTransform
    oneHotEncoding
    combineFeatures
    transformViaMappingDict
    transformViaFunction
    """
    
    # Example: Method transform a highly skewed data (ie one very large, one very small) to a more balanced plane
    # skewedFeatures = ['capital-gain', 'capital-loss']
    # featureY = myFeatureUtil.logTransform(features_raw, skewedFeatures)
    # myDataExplorationUtil.plotDistributionMultipleFields(skewedFeatures)
    def logTransform(self, featureOrig, skewedFields):
        featuresTransformed = pd.DataFrame(data=featureOrig)
        featuresTransformed[skewedFields] = featuresTransformed[skewedFields].apply(lambda x: np.log(x+1))
        return featuresTransformed
    
    # Example: Method normalized the numerical input into the an output between 0 and 1
    #  numericalFeatures = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    #  featureX = myFeatureUtil.minMaxScalerTransform(features_raw, numericalFeatures)
    def minMaxScalerTransform(self, featuresOrig, numericalFields):
        scaler = MinMaxScaler()
        featuresTransformed = pd.DataFrame(data=featuresOrig)
        featuresTransformed[numericalFields] = scaler.fit_transform(featuresTransformed[numericalFields])
        return featuresTransformed
    
    # Example: Method creates an one-hot encoding a field
    def oneHotEncodingTransform(self, featuresOrig, categoricalFields, prefixToAdd=''):
        featuresTransformed = pd.get_dummies(featuresOrig[categoricalFields], prefix=prefixToAdd)
        return featuresTransformed
    
    # Example: method combines a set of features
    #   myFeatureUtil.combineFeatures([featureOneHotEncoded, featureMinMaxTransformed, featureLogTransformed])
    def combineFeatures(self, features):
        return pd.concat(features)
    
    # Example: Method applies transforms a colum in df based on the definition of the mapping dictionary
    #   mappingDict = {'female': 0, 'male': 1, 0:0, 1:1}
    #   myFeatureUtil.transformViaMappingDict(data, 'Sex', mappingDict)
    def transformViaMappingDict(self, df, field, mappingDict):
        df[field] = data[field].map(mappingDict)
        return df
    
    # Example: Method applies transformation on fields in df via  a function. Notice the coupling between inputFieldsToFnc and func 
    #  outputField = 'Age'
    #  inputFieldsToFnc = ['Age', 'Pclass']
    #  def impute_age(cols): 
    #    age = cols[0]
    #    pclass = cols[1]
    #    if pd.isnull(age):
    #        if pclass == 1:
    #            return data[data['Pclass'] == 1]['Age'].median()
    #        else:
    #            return data[data['Pclass'] == 2]['Age'].median()
    #    else:
    #        return age
    # myFeatureUtil.transformViaFunction(data, outputField, inputFieldsToFnc, impute_age)
    def transformViaFunction(self, df, outputField, inputFieldsToFnc, func):
        data[outputField] = data[inputFieldsToFnc].apply(func, axis=1)
        return df
    
    # Method returns a PCA model which reduces the input feature dimension which maximizes capturing the variance in the data per principal component analysis iteration
    # Example usage: http://localhost:8888/notebooks/Udacity_connect/wk7/PCA-Solutions.ipynb
    #   good_data is our DF of features, which has been normalized and removedOutliers. type = pandas.core.frame.DataFrame
    #   pca: sklearn.decomposition.pca.PCA  = i.featureUtil.reduceFeatureDimensionViaPCA(features=good_data, numComponents=6)
    #   pca_samples: numpy.ndarray = pca.transform(log_samples)
    #   pca_results: DF = i.featureUtil.pca_results(good_data, pca)
    def reduceFeatureDimensionViaPCA(self, features, numComponents, originalFeatureHeight=None, originalFeatureWidth=None):
        print 'Extracting the top {} eigenfaces from {} faces'.format(numComponents, features.shape[0])
        t0 = time() # track time
        pca = PCA(n_components=numComponents, svd_solver='randomized', whiten=True).fit(features)
        print 'done in {:0.3f}s'.format(time() - t0)

        # Reshape the PCA components based on the image dimensions
        # Use this if you need the ORIGINAL coordinates
        featuresInOriginalDimensions = pca.components_.reshape((numComponents, originalFeatureHeight, originalFeatureWidth)) if (originalFeatureHeight != None) else None
        
        print 'Projecting the input data on the eigenfaces orthonormal basis'
        t0 = time()
        featuresPCA = pca.transform(features)
        print 'done in {:0.3f}s'.format(time() - t0)

        # print the explained variance
        topNComponents = 5
        for idx, var in enumerate(pca.explained_variance_ratio_[:topNComponents]):
            print 'Eigenface {} explains {:5.2f}% of the variance.'.format(idx+1, var*100.0)
        print '\nIn total, the first {} eigenfaces explain {:5.2f}% of the variance.'\
              .format(len(pca.explained_variance_ratio_),\
                      100.0*np.sum(pca.explained_variance_ratio_))

        # plot cumulative explained variance as a function of n_components
        plt.figure(figsize=(8, 6))
        var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100) # Cumulative Variance explains
        plt.plot(var1, color='b')
        plt.xlabel('n_components')
        plt.ylabel('Cumulative explained variance (%)')
        return pca
    
    # pca is of type sklearn.decomposition.pca.PCA, passed via reduceFeatureDimensionViaPCA
    def pca_results(self, df, pca):
        '''
        Create a DataFrame of the PCA results
        Includes dimension feature weights and explained variance
        Visualizes the PCA results
        '''

        # Dimension indexing
        dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

        # PCA components
        components = pd.DataFrame(np.round(pca.components_, 4), columns = df.keys())
        components.index = dimensions

        # PCA explained variance
        ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
        variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
        variance_ratios.index = dimensions

        # Create a bar plot visualization
        fig, ax = plt.subplots(figsize = (14,8))

        # Plot the feature weights as a function of the components
        components.plot(ax = ax, kind = 'bar');
        ax.set_ylabel("Feature Weights")
        ax.set_xticklabels(dimensions, rotation=0)

        # Display the explained variance ratios
        for i, ev in enumerate(pca.explained_variance_ratio_):
            ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

        # Return a concatenated DataFrame
        return pd.concat([variance_ratios, components], axis = 1)

class MyTrainingUtil:
    """
    Data split: Shuffle split
    Metrics: make_scorer
    Hyper-parameter tuning: Grid search
    Incorporate some code|references of the visual.py code base
    """
    
    # For regression
    def metricMeanSquare(self, yLabel, yPredict):
        return mean_squared_error(yLabel, yPredict)
    
    # Aka coefficient of determination.
    # Score tells the proportion of variance is y is explained by x.
    # Example: score of 0.40 means that 40 percent of the variance in Y is predictable from X
    # score = metric_r2([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
    # print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)
    def metricR2(self, yLabel, yPredict):
        return r2_score(yLabel, yPredict)
        
    # For classification
    # score = myTrainingUtil.metricAccuracy(y_train_label, y_predicted_by_model)
    def metricAccuracy(self, yLabel, yPredict):
        return accuracy_score(yLabel, yPredict)
    
    # For classification
    def metricConfusion(self, yLabel, yPredict):
        return confusion_matrix(yLabel, yPredict)
    
    # For classification
    # F1 score is a composite of both recall and precision
    def metricF1(self, yLabel, yPredict):
        return f1_score(yLabel, yPredict)
    
    # Outputs a report of recall, precision, and f1
    def reportClassification(self, yData, yPred, targetNames):
        # targetNames: aka classification class?
        print classification_report(yData, yPred, target_names=targetNames) 
    
    def trainTestSplit(self, data, label, testSize=0.2, randomState=21):
        xTrain, xTest, yTrain, yTest = train_test_split(data, label, test_size=testSize, random_state=randomState)
        return xTrain, xTest, yTrain, yTest
    
    def trainTestSplitForClassImbalances(self, data, label, testSize=0.2, randomState=21):
        xTrain, xTest, yTrain, yTest = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=randomState)
        return xTrain, xTest, yTrain, yTest

    # [ SIDE : DF vs panda.Series vs numpy.ndarray ]
    # data : DataFrame = pd.readCSV()
    # cols: Panda.Series = data[field]
    # output_horizonal_matrix: numpy.ndarray = data[field].values
    # output_vertical_matrix: numpy.ndarray = data[field].values.reshape(-1,1)
    # X_train: numpy.ndarray ,,,, = train_test_split(data)
    # lreg.fit(train_set['sqft_living'].values.reshape(-1,1), train_set['price'])

    
    # Example inputs: 
    #   tuningParams = {'max_depth': range(1,11), 'min_samples_leaf': range(1,2)}
    #   modelOptimized = myTrainingUtil.fitModel(x_train, y_train, DecisionTreeRegressor(), tuningParams, myTrainingUtil.metricR2)
    #   print "Optimized tuning parameters => {}".format(modelOptimized.get_params())
    # Good examples of looping through a couple of models: 
    #   http://localhost:8888/notebooks/projects/machine-learning/projects/finding_donors/finding_donors-FinalSolution-TOM.ipynb
    def fitModel(self, X, y, model, hyperparams, metric):
        
        # Create cross-validation sets from the training data
        cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

        # Transform 'performance_metric' into a scoring function using 'make_scorer' 
        scoring_fnc = make_scorer(metric)

        # Create the grid search cv object --> GridSearchCV()
        # Make sure to include the right parameters in the object:
        # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
        grid = GridSearchCV(estimator=model, param_grid=hyperparams, scoring=scoring_fnc, cv=cv_sets)

        # Fit the grid search object to the data to compute the optimal model
        grid = grid.fit(X, y)

        # Return the optimal model after fitting the data
        return grid.best_estimator_

    # Method outputs the tree decision in a graph
    # myTrainingUtil.plotTreeDecision(model, 'Titantic model')
    def plotTreeDecision(self, model, name):
        dot_data = tree.export_graphviz(clf1, out_file=None) 
        graph = graphviz.Source(dot_data) 
        graph.render("Titanic") 
        return graph
        
class MyPandaUtil:
    """
    Utility class for common panda operations
    """
    
    def diff2DF(self, df1, df2):
        merged = df1.merge(df2, indicator=True, how='outer')
        diff = merged[(merged['_merge'] == 'right_only') | (merged['_merge'] == 'left_only')]
        return diff

    def intersect2DF(self, df1, df2):
        merged = df1.merge(df2, indicator=True, how='inner')
        intersect = merged[merged['_merge'] == 'both']
        return intersect

    def concat2DF(self, a, b):
        frame = [a, b]
        return pd.concat(frame)
    
    def test(self):
        df1 = pd.DataFrame({'Buyer': ['Carl', 'Carl', 'Carl'], 'Quantity': [18, 3, 5, ]})
        df2 = pd.DataFrame({'Buyer': ['Carl', 'Mark', 'Carl', 'Carl'], 'Quantity': [2, 1, 18, 5]})
        common = self.intersect2DF(df1, df2)
        display('Intersect Elements:', common.head())

        diff = self.diff2DF(df1, df2)
        display('Common Elements:', diff.head())

        df3 = pd.DataFrame({'itemId':['1', '3']})
        df4 = pd.DataFrame({'itemId':['2']})
        concat = self.concat2DF(df3, df4)
        display('Concat 2 DFs:', concat.head())
        

# ------------------------------
# Instantiate helper classes
# ------------------------------
class Util:
    dataExplorationUtil = MyDataExplorationUtil()
    featureUtil = MyFeatureTransformUtil()
    trainingUtil = MyTrainingUtil()
    pandaUtil = MyPandaUtil()
    dataCleaningUtil = MyDataCleaningUtil()
util = Util()


