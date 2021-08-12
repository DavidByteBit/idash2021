# date: July 22, 2021
# name: Martine De Cock
# description: Training ML models on IDASH2021, Track 3 data

# DP LR

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
from numpy import savetxt
import sys

rows = None
cols = None
num_of_folds = int(sys.argv[1])
if len(sys.argv) > 2:
    rows = int(sys.argv[2])
    cols = int(sys.argv[3])



##################################################################################

def preprocess(dirty_df):
    dirty_df = dirty_df.drop(['patient_id', 'cohort_type'], axis=1)
    target_map = {u'1': 1, u'0': 0}
    dirty_df['__target__'] = dirty_df['cohort_flag'].map(str).map(target_map)
    dirty_df = dirty_df.drop(['cohort_flag'], axis=1)
    clean_X = dirty_df.drop('__target__', axis=1)

    if cols is not None:
        clean_X = clean_X.iloc[:, :cols]

    clean_X = clean_X.to_numpy()
    clean_X = preprocessing.normalize(clean_X, norm='l2')
    clean_y = np.array(dirty_df['__target__'])

    if rows is not None:
        return clean_X[0:rows], clean_y[0:rows]

    return clean_X, clean_y


##################################################################################


# Load the data
df1 = pd.read_csv('data/alice_data.csv')
df2 = pd.read_csv('data/bob_data.csv')

print(df1.shape)

X1, y1 = preprocess(df1)
X2, y2 = preprocess(df2)

# This will hold 4 accuracy results for each of 5 folds
# (1) accuracy of model trained on data from P1
# (2) accuracy of model trained on data from P2
# (3) accuracy of model trained on data from P1 and from P2
# (4) accuracy of aggregation of model (1) and (2) from above
LRresults = np.zeros((5, 5))

kf1 = KFold(n_splits=num_of_folds, shuffle=True, random_state=42)
kf2 = KFold(n_splits=num_of_folds, shuffle=True, random_state=42)

epsilon = 1
mylambda = 0.5

fold = 0

for (train1_indices, test1_indices), (train2_indices, test2_indices) in zip(kf1.split(X1, y1), kf2.split(X2, y2)):

    print("Starting to process fold {n}".format(n=fold))

    AliceX_train, AliceX_test = X1[train1_indices, :].tolist(), X1[test1_indices, :].tolist()
    Alicey_train, Alicey_test = y1[train1_indices].tolist(), y1[test1_indices].tolist()
    BobX_train, BobX_test = X2[train2_indices, :].tolist(), X2[test2_indices, :].tolist()
    Boby_train, Boby_test = y2[train2_indices].tolist(), y2[test2_indices].tolist()

    # Get rid of scientific notation
    AliceX_train = [[str(f'{j:.10f}') for j in i] for i in AliceX_train]
    AliceX_test = [[str(f'{j:.10f}') for j in i] for i in AliceX_test]
    Alicey_train = [str(f'{i:.10f}') for i in Alicey_train]
    Alicey_test = [str(f'{i:.10f}') for i in Alicey_test]

    BobX_train = [[str(f'{j:.10f}') for j in i] for i in BobX_train]
    BobX_test = [[str(f'{j:.10f}') for j in i] for i in BobX_test]
    Boby_train = [str(f'{i:.10f}') for i in Boby_train]
    Boby_test = [str(f'{i:.10f}') for i in Boby_test]

    savetxt('data/Alice/train_X_fold{n}.csv'.format(n=fold), AliceX_train, delimiter=',', fmt='%s')
    savetxt('data/Alice/train_y_fold{n}.csv'.format(n=fold), Alicey_train, delimiter=',', fmt='%s')
    savetxt('data/Alice/test_X_fold{n}.csv'.format(n=fold), AliceX_test, delimiter=',', fmt='%s')
    savetxt('data/Alice/test_y_fold{n}.csv'.format(n=fold), Alicey_test, delimiter=',', fmt='%s')

    savetxt('data/Bob/train_X_fold{n}.csv'.format(n=fold), BobX_train, delimiter=',', fmt='%s')
    savetxt('data/Bob/train_y_fold{n}.csv'.format(n=fold), Boby_train, delimiter=',', fmt='%s')
    savetxt('data/Bob/test_X_fold{n}.csv'.format(n=fold), BobX_test, delimiter=',', fmt='%s')
    savetxt('data/Bob/test_y_fold{n}.csv'.format(n=fold), Boby_test, delimiter=',', fmt='%s')

    fold += 1
