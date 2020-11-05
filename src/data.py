import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def DataMetrics(dataframe):
    '''
    Returns information on dataframe to be analyzed.
    Input: Panda Dataframe
    Output: Null
    '''

    num_features = dataframe.shape[1]

    print( '\n'.join([
        f'Total features: {num_features}',
        f'File features: {dataframe.columns.values}',
        f'Parametric Description',
        f'{dataframe.describe()}'
        ]))


def MinMaxNormalization(dataframe,col1,col2):
    '''
    Normalize the pressure columns in the dataframe based on MinMaxScaler and
    converts binary state from integer to float to be fed into Neural Network.
    Input: Dataframe
           Dataframe Column 1
           Dataframe Column 2
    Output: Dataframe (normalized)
    '''

    p1 = dataframe[[col1]].values.astype(float)
    p2 = dataframe[[col2]].values.astype(float)

    # Create a min max processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    norm_p1 = min_max_scaler.fit_transform(p1)
    norm_p2 = min_max_scaler.fit_transform(p2)

    dataframe[col1] = norm_p1
    dataframe[col2] = norm_p2

    dataframe['input_state'] = dataframe['input_state'].astype(float)
    dataframe['feedback_state'] = dataframe['feedback_state'].astype(float)

    print('Normalzied Dataframe returned...')
    print(dataframe.head(5))

    return dataframe

def OneShot(dataframe,column):
    '''
    Turn the label column specified into one shot encoding. The function will read
    the size of the column and the unique values in order to create the oneshot 
    encoding.
    Input: Dataframe
           Column String to be used as labels
    Output: Input Dataframe
            Labels Dataframe
    '''

    indices = len(dataframe[column])
    depth = pd.Series(dataframe[column].values).unique()

    print("Total number of labels: {}".format(indices))
    print("Unique number of labels: {}".format(depth))

    label_dataframe = pd.get_dummies(dataframe[column].values)

    dataframe = dataframe.drop([column],axis=1)

    print('Input Dataframe Shape: {}'.format(dataframe.shape))
    print('Target Dataframe Shape: {}'.format(label_dataframe.shape))

    return dataframe,label_dataframe


def ModelSplit(dataframe, label_dataframe,split):
    '''
    Split the two returned dataframes into a train and test dataframe.
    Input: Input Dataframe
           Labels Dataframe
           Percent of Test Labels as float
    Output: Traning Input and Label Array
            Testing Input and Label Array
    '''
    
    df = dataframe.to_numpy()
    lb_df = label_dataframe.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(df, lb_df, test_size=split)

    print('Data split into training and test arrays...')
    print('Shape of X Train: {}'.format(X_train.shape))
    print('Shape of X Test: {}'.format(X_test.shape))
    print('Shape of y Train: {}'.format(y_train.shape))
    print('Shape of y Train: {}'.format(y_test.shape))

    return X_train, X_test, y_train, y_test
