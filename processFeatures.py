import numpy as np
import pandas
import functools

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def extractFeatures(inputfile, outputfile):
    dataset = pandas.read_csv(inputfile)
    floatFeaturesList = ['rally',
                         'net.clearance',
                         'previous.net.clearance',
                         'speed',
                         'previous.speed',
                         'distance.from.sideline',
                         'previous.distance.from.sideline',
                         'depth',
                         'previous.depth',
                         'opponent.depth',
                         'player.distance.travelled',
                         'player.impact.depth',
                         'player.impact.distance.from.center',
                         'player.depth',
                         'player.distance.from.center',
                         'opponent.distance.from.center',
                         'previous.time.to.net']
    binaryFeaturesList = ['outside.sideline',
                          'outside.baseline',
                          'same.side',
                          'server.is.impact.player',
                          'serve'] # 1st serve = 0, 2nd serve = 1
    categoryFeaturesList = ['hitpoint0',
                            'hitpoint1',
                            'hitpoint2',
                            'hitpoint3',
                            'previous.hitpoint0',
                            'previous.hitpoint1',
                            'previous.hitpoint2',
                            'previous.hitpoint3']
    
    from scipy import stats
    for col in ['distance.from.sideline',
                'depth',
                'player.impact.distance.from.center',
                'player.depth',
                'player.distance.from.center',
                'previous.speed',
                'previous.net.clearance',
                'previous.distance.from.sideline',
                'previous.depth',
                'opponent.distance.from.center',
                'player.impact.depth',
                'previous.time.to.net']:
        xT, _ = stats.boxcox(dataset[[col]].values)
        dataset[[col]] = xT
        
    X = dataset[floatFeaturesList].values.astype(float)

    # Scale features so that data is centered around mean with unit variance.
    X = preprocessing.scale(X)

    # Binary features.
    dataset[['serve']] = preprocessing.binarize(dataset[['serve']].values, threshold=1.5, copy=False)
    X_binary = dataset[binaryFeaturesList].values.astype(int)
    X = np.append(X, X_binary, axis = 1)

    # Convert category features to one hot encoded features.
    le = LabelEncoder()
    le.fit(['B', 'F', 'U', 'V'])
    ohEnc = OneHotEncoder()
    for col in ['hitpoint', 'previous.hitpoint']:
        dataset[col] = le.transform(dataset[col])
        ohEnc.fit(dataset[[col]])
        temp = ohEnc.transform(dataset[[col]])
        X = np.append(X, temp.toarray(), axis = 1)

    Xdf = pandas.DataFrame(X)
    Xdf['outcome']  = dataset['outcome']
    colNames = np.append(floatFeaturesList, binaryFeaturesList, axis = 0)
    colNames = np.append(colNames, categoryFeaturesList, axis = 0)
    colNames = np.append(colNames, ['outcome'], axis = 0)
    Xdf.columns = colNames
    Xdf.to_csv(outputfile, index = False)


def main():
    extractFeatures('mens_train_file.csv', 'mens_train_features.csv')
    extractFeatures('womens_train_file.csv', 'womens_train_features.csv')

    extractFeatures('mens_test_file_reorder.csv', 'mens_test_features.csv')
    extractFeatures('womens_test_file_reorder.csv', 'womens_test_features.csv')

if __name__ == "__main__":
    main()


    
    #from sklearn.preprocessing import MinMaxScaler
    #min_max = MinMaxScaler()
    ## Scaling down both train and test data set
    #X = min_max.fit_transform(X)