import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

df_train = pd.read_csv('D_train.csv')
df_train.drop('Unnamed: 0', axis=1, inplace=True)

###########################################################
# Group by users
###########################################################
df_user = []
for each_user in np.unique(df_train['User']):
    df_user.append(df_train[df_train['User'] == each_user].drop('User', axis=1).reset_index().drop('index', axis=1))
    
###########################################################
# Create permutation invariant features
# design feature HERE
###########################################################

df_user_num = []
df_user_X = []
df_user_Y = []
df_user_Z = []
df_user_square_root = []
df_user_mul = []
df_user_cross_mul = []

for each_df in df_user:
    
    # Given by tutorial
    
    each_df_num = each_df[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]
    each_df_num['Num'] = each_df_num.count(axis=1) 
    each_df_num.drop(['X0','X1','X2','X3','X4','X5','X6','X7','X8', 'X9','X10','X11'], axis=1, inplace=True)
    df_user_num.append(each_df_num.copy())
    
    each_df_X = each_df[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]
    each_df_X['XMean'] = each_df_X[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].mean(axis=1, skipna=True)
    each_df_X['XStd'] = each_df_X[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].std(axis=1, skipna=True)
    each_df_X['XMax'] = each_df_X[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].max(axis=1, skipna=True)
    each_df_X['XMin'] = each_df_X[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].min(axis=1, skipna=True)
    each_df_X.drop(['X0','X1','X2','X3','X4','X5','X6','X7','X8', 'X9','X10','X11'], axis=1, inplace=True)
    df_user_X.append(each_df_X.copy())
    
    each_df_Y = each_df[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']]
    each_df_Y['YMean'] = each_df_Y[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']].mean(axis=1, skipna=True)
    each_df_Y['YStd'] = each_df_Y[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']].std(axis=1, skipna=True)
    each_df_Y['YMax'] = each_df_Y[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']].max(axis=1, skipna=True)
    each_df_Y['YMin'] = each_df_Y[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']].min(axis=1, skipna=True)
    each_df_Y.drop(['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11'], axis=1, inplace=True)
    df_user_Y.append(each_df_Y.copy())
    
    each_df_Z = each_df[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']]
    each_df_Z['ZMean'] = each_df_Z[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']].mean(axis=1, skipna=True)
    each_df_Z['ZStd'] = each_df_Z[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']].std(axis=1, skipna=True)
    each_df_Z['ZMax'] = each_df_Z[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']].max(axis=1, skipna=True)
    each_df_Z['ZMin'] = each_df_Z[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']].min(axis=1, skipna=True)
    each_df_Z.drop(['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11'], axis=1, inplace=True)
    df_user_Z.append(each_df_Z.copy())
    
    # (1/n) * sqrt(x2 + y2 + z2)
    
    each_df_SR_0_s = (each_df['X0'].apply(np.square) + each_df['Y0'].apply(np.square) + each_df['Z0'].apply(np.square)).apply(np.sqrt)
    each_df_SR_1_s = (each_df['X1'].apply(np.square) + each_df['Y1'].apply(np.square) + each_df['Z1'].apply(np.square)).apply(np.sqrt)
    each_df_SR_2_s = (each_df['X2'].apply(np.square) + each_df['Y2'].apply(np.square) + each_df['Z2'].apply(np.square)).apply(np.sqrt)
    each_df_SR_3_s = (each_df['X3'].apply(np.square) + each_df['Y3'].apply(np.square) + each_df['Z3'].apply(np.square)).apply(np.sqrt)
    each_df_SR_4_s = (each_df['X4'].apply(np.square) + each_df['Y4'].apply(np.square) + each_df['Z4'].apply(np.square)).apply(np.sqrt)
    each_df_SR_5_s = (each_df['X5'].apply(np.square) + each_df['Y5'].apply(np.square) + each_df['Z5'].apply(np.square)).apply(np.sqrt)
    each_df_SR_6_s = (each_df['X6'].apply(np.square) + each_df['Y6'].apply(np.square) + each_df['Z6'].apply(np.square)).apply(np.sqrt)
    each_df_SR_7_s = (each_df['X7'].apply(np.square) + each_df['Y7'].apply(np.square) + each_df['Z7'].apply(np.square)).apply(np.sqrt)
    each_df_SR_8_s = (each_df['X8'].apply(np.square) + each_df['Y8'].apply(np.square) + each_df['Z8'].apply(np.square)).apply(np.sqrt)
    each_df_SR_9_s = (each_df['X9'].apply(np.square) + each_df['Y9'].apply(np.square) + each_df['Z9'].apply(np.square)).apply(np.sqrt)
    each_df_SR_10_s = (each_df['X10'].apply(np.square) + each_df['Y10'].apply(np.square) + each_df['Z10'].apply(np.square)).apply(np.sqrt)
    each_df_SR_11_s = (each_df['X11'].apply(np.square) + each_df['Y11'].apply(np.square) + each_df['Z11'].apply(np.square)).apply(np.sqrt)
    each_df_SR_frame = pd.DataFrame({0:each_df_SR_0_s, 1:each_df_SR_1_s, 2:each_df_SR_2_s, 3:each_df_SR_3_s, 4:each_df_SR_4_s,
                                    5:each_df_SR_5_s, 6:each_df_SR_6_s, 7:each_df_SR_7_s, 8:each_df_SR_8_s, 9:each_df_SR_9_s,
                                    10:each_df_SR_10_s, 11:each_df_SR_11_s})
    each_df_SR_frame['SR'] = each_df_SR_frame.mean(axis=1, skipna=True)
    each_df_SR_frame.drop([0,1,2,3,4,5,6,7,8,9,10,11], axis=1, inplace=True)
    df_user_square_root.append(each_df_SR_frame)
    
    # (1/n) cub(xyz)  mul
    
    f = lambda x: np.sign(x) * np.power(abs(x), 1./3)
    
    each_df_mul_0_s = (each_df['X0'] * each_df['Y0'] * each_df['Z0']).apply(f)
    each_df_mul_1_s = (each_df['X1'] * each_df['Y1'] * each_df['Z1']).apply(f)
    each_df_mul_2_s = (each_df['X2'] * each_df['Y2'] * each_df['Z2']).apply(f)
    each_df_mul_3_s = (each_df['X3'] * each_df['Y3'] * each_df['Z3']).apply(f)
    each_df_mul_4_s = (each_df['X4'] * each_df['Y4'] * each_df['Z4']).apply(f)
    each_df_mul_5_s = (each_df['X5'] * each_df['Y5'] * each_df['Z5']).apply(f)
    each_df_mul_6_s = (each_df['X6'] * each_df['Y6'] * each_df['Z6']).apply(f)
    each_df_mul_7_s = (each_df['X7'] * each_df['Y7'] * each_df['Z7']).apply(f)
    each_df_mul_8_s = (each_df['X8'] * each_df['Y8'] * each_df['Z8']).apply(f)
    each_df_mul_9_s = (each_df['X9'] * each_df['Y9'] * each_df['Z9']).apply(f)
    each_df_mul_10_s = (each_df['X10'] * each_df['Y10'] * each_df['Z10']).apply(f)
    each_df_mul_11_s = (each_df['X11'] * each_df['Y11'] * each_df['Z11']).apply(f)
    each_df_mul_frame = pd.DataFrame({0:each_df_mul_0_s, 1:each_df_mul_1_s, 2:each_df_mul_2_s, 3:each_df_mul_3_s, 4:each_df_mul_4_s,
                                    5:each_df_mul_5_s, 6:each_df_mul_6_s, 7:each_df_mul_7_s, 8:each_df_mul_8_s, 9:each_df_mul_9_s,
                                    10:each_df_mul_10_s, 11:each_df_mul_11_s})
    each_df_mul_frame['Mul'] = each_df_mul_frame.mean(axis=1, skipna=True)
    each_df_mul_frame.drop([0,1,2,3,4,5,6,7,8,9,10,11], axis=1, inplace=True)
    df_user_mul.append(each_df_mul_frame)
    
    # (1/n) sqrt(|xy + yz + xz|)  cross mul
    
    each_df_CM_0_s = (each_df['X0'] * each_df['Y0'] + each_df['Y0'] * each_df['Z0'] + each_df['X0'] * each_df['Z0']).apply(np.abs).apply(np.sqrt)
    each_df_CM_1_s = (each_df['X1'] * each_df['Y1'] + each_df['Y1'] * each_df['Z1'] + each_df['X1'] * each_df['Z1']).apply(np.abs).apply(np.sqrt)
    each_df_CM_2_s = (each_df['X2'] * each_df['Y2'] + each_df['Y2'] * each_df['Z2'] + each_df['X2'] * each_df['Z2']).apply(np.abs).apply(np.sqrt)
    each_df_CM_3_s = (each_df['X3'] * each_df['Y3'] + each_df['Y3'] * each_df['Z3'] + each_df['X3'] * each_df['Z3']).apply(np.abs).apply(np.sqrt)
    each_df_CM_4_s = (each_df['X4'] * each_df['Y4'] + each_df['Y4'] * each_df['Z4'] + each_df['X4'] * each_df['Z4']).apply(np.abs).apply(np.sqrt)
    each_df_CM_5_s = (each_df['X5'] * each_df['Y5'] + each_df['Y5'] * each_df['Z5'] + each_df['X5'] * each_df['Z5']).apply(np.abs).apply(np.sqrt)
    each_df_CM_6_s = (each_df['X6'] * each_df['Y6'] + each_df['Y6'] * each_df['Z6'] + each_df['X6'] * each_df['Z6']).apply(np.abs).apply(np.sqrt)
    each_df_CM_7_s = (each_df['X7'] * each_df['Y7'] + each_df['Y7'] * each_df['Z7'] + each_df['X7'] * each_df['Z7']).apply(np.abs).apply(np.sqrt)
    each_df_CM_8_s = (each_df['X8'] * each_df['Y8'] + each_df['Y8'] * each_df['Z8'] + each_df['X8'] * each_df['Z8']).apply(np.abs).apply(np.sqrt)
    each_df_CM_9_s = (each_df['X9'] * each_df['Y9'] + each_df['Y9'] * each_df['Z9'] + each_df['X9'] * each_df['Z9']).apply(np.abs).apply(np.sqrt)
    each_df_CM_10_s = (each_df['X10'] * each_df['Y10'] + each_df['Y10'] * each_df['Z10'] + each_df['X10'] * each_df['Z10']).apply(np.abs).apply(np.sqrt)
    each_df_CM_11_s = (each_df['X11'] * each_df['Y11'] + each_df['Y11'] * each_df['Z11'] + each_df['X11'] * each_df['Z11']).apply(np.abs).apply(np.sqrt)
    each_df_CM_frame = pd.DataFrame({0:each_df_CM_0_s, 1:each_df_CM_1_s, 2:each_df_CM_2_s, 3:each_df_CM_3_s, 4:each_df_CM_4_s,
                                    5:each_df_CM_5_s, 6:each_df_CM_6_s, 7:each_df_CM_7_s, 8:each_df_CM_8_s, 9:each_df_CM_9_s,
                                    10:each_df_CM_10_s, 11:each_df_CM_11_s})
    each_df_CM_frame['CM'] = each_df_CM_frame.mean(axis=1, skipna=True)
    each_df_CM_frame.drop([0,1,2,3,4,5,6,7,8,9,10,11], axis=1, inplace=True)
    df_user_cross_mul.append(each_df_CM_frame)
    
# Balance the dataset or not
num_each_class = np.zeros(5)
for each in df_user:
    for each_class in each['Class']:
        num_each_class[each_class - 1] += 1
print('Number of each class before balance', num_each_class, 'Number of total samples', np.sum(num_each_class))

df_user_total = []

for i in range(len(df_user_num)):
    tmp = pd.concat([df_user[i]['Class'], df_user_num[i], df_user_X[i], df_user_Y[i], df_user_Z[i], df_user_square_root[i], df_user_mul[i], df_user_cross_mul[i]],axis=1)
    df_user_total.append(tmp.copy())


###########################################################
# Different data balancing method, comment / uncomment to 
# show the result of upsampling and downsampling
###########################################################

data_augmentor=SMOTE(random_state=0)
# data_augmentor=RandomUnderSampler(random_state=0)

df_user_total_balanced = []
for each in df_user_total:
    x_resample, y_resample=data_augmentor.fit_sample(each.drop('Class', axis=1), each['Class'])
    x_resample['Class'] = y_resample
    cols = list(x_resample)
    cols.insert(0,cols.pop(cols.index('Class')))    
    x_resample = x_resample.loc[:,cols]
    df_user_total_balanced.append(x_resample)

num_each_class = np.zeros(5)
for each in df_user_total_balanced: # balanced
    for each_class in each['Class']:
        num_each_class[each_class - 1] += 1
print(num_each_class, np.sum(num_each_class))

###########################################################
# Show the number in the training set and validation 
# set in different iterations.
###########################################################

total_numb = 0
print('Show the number in the training set and validation set in different iterations.')
for each in df_user_total_balanced:
    total_numb += len(each)
for each in df_user_total_balanced:
    print("Training set", (total_numb - len(each)), 'Test set', len(each))
    
###########################################################
# Show the result of different normalization methods
# also effect of Phi machine is tested here
# comment / uncomment to show different results
###########################################################
    
normization = StandardScaler()
# normization = MinMaxScaler()

train_score_NB = []
val_score_NB = []

# Naive Bayes               ALL
for i in range(len(df_user_total_balanced)):
    # balanced
    df_user_train = df_user_total_balanced[:i] + df_user_total_balanced[i+1:]
    df_user_val = df_user_total_balanced[i]
    train_df =[]
    val_df = df_user_val.copy()
    for each_subset in df_user_train:
        if isinstance(train_df, list):
            train_df = each_subset
        else:
            train_df = pd.concat((train_df, each_subset), axis=0, ignore_index=True)
            
    # Finished train test split
    normization.fit(train_df.drop('Class', axis=1))
    train_data = normization.transform(train_df.drop('Class', axis=1))
    train_label = np.array(train_df['Class'])
    
    val_data = normization.transform(val_df.drop('Class', axis=1))
    val_label = np.array(val_df['Class'])
    # Phi machine
    poly = PolynomialFeatures(2, interaction_only=False)
    train_data_poly = poly.fit_transform(train_data)
    val_data_poly = poly.fit_transform(val_data)
    # classify
    gnb = GaussianNB()
    
    # Phi machine
#     gnb.fit(train_data_poly, train_label) # Phi machine
#     train_score_NB.append(gnb.score(train_data_poly, train_label))
#     val_score_NB.append(gnb.score(val_data_poly, val_label))
    
    gnb.fit(train_data, train_label) # Linear machine
    train_score_NB.append(gnb.score(train_data, train_label))
    val_score_NB.append(gnb.score(val_data, val_label))

print('train score average is', np.mean(train_score_NB))
print('validation score average is', np.mean(val_score_NB))
print('train score standard var is', np.std(train_score_NB))
print('validation score standard var is', np.std(val_score_NB))


###########################################################
# Feature drop method
###########################################################
train_col = train_df.columns
train_score_drop = []
train_std_drop = []
val_score_drop = []
val_std_drop = []

for each_col in train_col:
    if each_col == 'Class':
        continue
    train_score_NB = []
    val_score_NB = []

    # Naive Bayes       ALL
    for i in range(len(df_user_total_balanced)):
        # balanced
        df_user_train = df_user_total_balanced[:i] + df_user_total_balanced[i+1:]
        df_user_val = df_user_total_balanced[i]
        train_df =[]
        val_df = df_user_val.copy()
        for each_subset in df_user_train:
            if isinstance(train_df, list):
                train_df = each_subset
            else:
                train_df = pd.concat((train_df, each_subset), axis=0, ignore_index=True)

        # Finished train test split
        normization.fit(train_df.drop(['Class'] + [each_col], axis=1))
        train_data = normization.transform(train_df.drop(['Class'] + [each_col], axis=1))
        train_label = np.array(train_df['Class'])

        val_data = normization.transform(val_df.drop(['Class'] + [each_col], axis=1))
        val_label = np.array(val_df['Class'])
        # Phi machine
        poly = PolynomialFeatures(2)
        train_data_poly = poly.fit_transform(train_data)

        # classify
        gnb = GaussianNB()
        gnb.fit(train_data, train_label) # Linear machine
        train_score_NB.append(gnb.score(train_data, train_label))
        val_score_NB.append(gnb.score(val_data, val_label))


    train_score_drop.append(np.mean(train_score_NB))
    train_std_drop.append(np.std(train_score_NB))
    val_score_drop.append(np.mean(val_score_NB))
    val_std_drop.append(np.std(val_score_NB))
for i in range(len(val_score_drop)):
    print('Dropped Features:',train_col[i + 1], '\nTraining Accuracy', train_score_drop[i],
         '\nValidation Accuracy:',val_score_drop[i])
         

###########################################################
# SVM cross validation parameter selection part
###########################################################
size_C = 10
size_gamma = 10
C = np.logspace(-3, 3, num=size_C)
gamma = np.logspace(-3, 3, num=size_gamma)
ACC_SVM = np.zeros((size_gamma, size_C))
DEV_SVM = np.zeros((size_gamma, size_C))
train_score_SVM = []
val_score_SVM = []
for i in range(size_gamma):
    for j in range(size_C):
        print(i, j)
        svmclf = svm.SVC(C=C[j], kernel='rbf', gamma=gamma[i])
        acc = []
        
        for k in range(len(df_user_total_balanced)):
            # balanced
            df_user_train = df_user_total_balanced[:k] + df_user_total_balanced[k+1:]
            df_user_val = df_user_total_balanced[k]
            train_df = []
            val_df = df_user_val.copy()
            for each_subset in df_user_train:
                if isinstance(train_df, list):
                    train_df = each_subset
                else:
                    train_df = pd.concat((train_df, each_subset), axis=0, ignore_index=True)
                    
            # Finished train test split
            normization.fit(train_df.drop(['Class', 'Mul'], axis=1))
            train_data = normization.transform(train_df.drop(['Class', 'Mul'], axis=1))
            train_label = np.array(train_df['Class'])

            val_data = normization.transform(val_df.drop(['Class', 'Mul'], axis=1))
            val_label = np.array(val_df['Class'])
            # classify
            svmclf.fit(train_data, train_label) # Linear machine
            acc.append(svmclf.score(val_data, val_label))
        ACC_SVM[i, j] = np.mean(acc)
        DEV_SVM[i, j] = np.std(acc)
plt.imshow(ACC_SVM)
plt.colorbar()

# Show the result

max_pos = np.where(ACC_SVM==np.amax(ACC_SVM))
if np.size(max_pos) != 2:
    num_same = len(max_pos[0])
    min_std = np.Inf
    for i in range(num_same):
        if ACC_SVM[max_pos[0][i], max_pos[1][i]] < min_std:
            min_std = ACC_SVM[max_pos[0][i], max_pos[1][i]]
            temp = (max_pos[0][i], max_pos[1][i])
    final_pos = temp
else:
    final_pos = max_pos
print('Max average accuracy is', ACC_SVM[final_pos])
print('Corresponding standard deviation is', DEV_SVM[final_pos])
print('Corresponding gamma is', gamma[final_pos[0]])
print('Corresponding C is', C[final_pos[1]])

###########################################################
# Perceptron cross validation parameter selection part
###########################################################
size_eta0 = 20
eta0 = np.logspace(-5, 3, num=size_eta0)
ACC_PER = np.zeros(size_eta0)
DEV_PER = np.zeros(size_eta0)
train_score_PER = []
val_score_PER = []
for i in range(size_eta0):
    print(i)
    perclf = Perceptron(eta0=eta0[i])
    acc = []
    for k in range(len(df_user_total_balanced)):
        # balanced
        df_user_train = df_user_total_balanced[:k] + df_user_total_balanced[k+1:]
        df_user_val = df_user_total_balanced[k]
        train_df = []
        val_df = df_user_val.copy()
        for each_subset in df_user_train:
            if isinstance(train_df, list):
                train_df = each_subset
            else:
                train_df = pd.concat((train_df, each_subset), axis=0, ignore_index=True)
        # Finished train test split
        normization.fit(train_df.drop(['Class', 'Mul'], axis=1))
        train_data = normization.transform(train_df.drop(['Class', 'Mul'], axis=1))
        train_label = np.array(train_df['Class'])

        val_data = normization.transform(val_df.drop(['Class', 'Mul'], axis=1))
        val_label = np.array(val_df['Class'])
        perclf.fit(train_data, train_label) # Linear machine
        acc.append(perclf.score(val_data, val_label))
            
    ACC_PER[i] = np.mean(acc)
    DEV_PER[i] = np.std(acc)
per_final_eta = eta0[np.argmax(ACC_PER)]
for i in range(len(ACC_PER)):
    print(eta0[i], ACC_PER[i])
    
fig = plt.figure()
plt.plot(eta0, ACC_PER)
plt.xscale('log')
plt.title('Perceptron')
plt.xlabel('eta')
plt.ylabel('mean accuracy')


###########################################################
# KNN cross validation parameter selection part
###########################################################
size_nei = 50
nei = np.linspace(2, 100, num=size_nei)
ACC_KNN = np.zeros(size_nei)
DEV_KNN = np.zeros(size_nei)
train_score_KNN = []
val_score_KNN = []
for i in range(size_nei):
    print(i)
    knnclf = KNeighborsClassifier(n_neighbors=int(nei[i]))
    acc = []
    for k in range(len(df_user_total_balanced)):
        # balanced
        df_user_train = df_user_total_balanced[:k] + df_user_total_balanced[k+1:]
        df_user_val = df_user_total_balanced[k]
        train_df = []
        val_df = df_user_val.copy()
        for each_subset in df_user_train:
            if isinstance(train_df, list):
                train_df = each_subset
            else:
                train_df = pd.concat((train_df, each_subset), axis=0, ignore_index=True)
        # Finished train test split
        normization.fit(train_df.drop(['Class', 'Mul'], axis=1))
        train_data = normization.transform(train_df.drop(['Class', 'Mul'], axis=1))
        train_label = np.array(train_df['Class'])

        val_data = normization.transform(val_df.drop(['Class', 'Mul'], axis=1))
        val_label = np.array(val_df['Class'])
        # classify

        knnclf.fit(train_data, train_label) # Linear machine
        acc.append(knnclf.score(val_data, val_label))
            
    ACC_KNN[i] = np.mean(acc)
    DEV_KNN[i] = np.std(acc)

knn_final_nei = nei[np.argmax(ACC_KNN)]
for i in range(len(ACC_KNN)):
    print(nei[i], ACC_KNN[i])
fig = plt.figure()
plt.plot(nei, ACC_KNN)
plt.title('KNN')
plt.xlabel('Number of neighborhood')
plt.ylabel('mean accuracy')

###########################################################
# Naive Bayes cross validation
###########################################################

nbclf = GaussianNB()
acc_nb = []
for k in range(len(df_user_total_balanced)):
    # balanced
    df_user_train = df_user_total_balanced[:k] + df_user_total_balanced[k+1:]
    df_user_val = df_user_total_balanced[k]
    train_df = []
    val_df = df_user_val.copy()
    for each_subset in df_user_train:
        if isinstance(train_df, list):
            train_df = each_subset
        else:
            train_df = pd.concat((train_df, each_subset), axis=0, ignore_index=True)
    # Finished train test split
    normization.fit(train_df.drop(['Class', 'Mul'], axis=1))
    train_data = normization.transform(train_df.drop(['Class', 'Mul'], axis=1))
    train_label = np.array(train_df['Class'])

    val_data = normization.transform(val_df.drop(['Class', 'Mul'], axis=1))
    val_label = np.array(val_df['Class'])

    # classify
    nbclf.fit(train_data, train_label) # Linear machine
    acc_nb.append(nbclf.score(val_data, val_label))

print('mean_val: nb, svm, per, knn:', np.mean(acc_nb), ACC_SVM[final_pos[0], final_pos[1]][0], ACC_PER[np.argmax(ACC_PER)], ACC_KNN[np.argmax(ACC_KNN)])
print('std_val: nb, svm, per, knn:', np.std(acc_nb), DEV_SVM[final_pos[0], final_pos[1]][0], DEV_PER[np.argmax(ACC_PER)], DEV_KNN[np.argmax(ACC_KNN)])


###########################################################
# Processing test set and using the whole training set
###########################################################

df_test = pd.read_csv('D_test.csv')
df_test.drop('Unnamed: 0', axis=1, inplace=True)
df_user12 = df_test[df_test['User'] == 12].drop('User', axis=1).reset_index().drop('index', axis=1)
df_user13 = df_test[df_test['User'] == 13].drop('User', axis=1).reset_index().drop('index', axis=1)
df_user14 = df_test[df_test['User'] == 14].drop('User', axis=1).reset_index().drop('index', axis=1)
df_user_test = []
df_user_test.append(df_user12)
df_user_test.append(df_user13)
df_user_test.append(df_user14)

df_user_num_test = []
df_user_X_test = []
df_user_Y_test = []
df_user_Z_test = []
df_user_square_root_test = []
df_user_mul_test = []
df_user_cross_mul_test = []

def power_3(x):
    return np.power(x, 1/3)

for each_df in df_user_test:
    
    # Given by tutorial
    each_df_num = each_df[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]
    each_df_num['Num'] = each_df_num.count(axis=1) 
    each_df_num.drop(['X0','X1','X2','X3','X4','X5','X6','X7','X8', 'X9','X10','X11'], axis=1, inplace=True)
    df_user_num_test.append(each_df_num.copy())
    
    each_df_X = each_df[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]
    each_df_X['XMean'] = each_df_X[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].mean(axis=1, skipna=True)
    each_df_X['XStd'] = each_df_X[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].std(axis=1, skipna=True)
    each_df_X['XMax'] = each_df_X[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].max(axis=1, skipna=True)
    each_df_X['XMin'] = each_df_X[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].min(axis=1, skipna=True)
    each_df_X.drop(['X0','X1','X2','X3','X4','X5','X6','X7','X8', 'X9','X10','X11'], axis=1, inplace=True)
    df_user_X_test.append(each_df_X.copy())
    
    each_df_Y = each_df[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']]
    each_df_Y['YMean'] = each_df_Y[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']].mean(axis=1, skipna=True)
    each_df_Y['YStd'] = each_df_Y[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']].std(axis=1, skipna=True)
    each_df_Y['YMax'] = each_df_Y[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']].max(axis=1, skipna=True)
    each_df_Y['YMin'] = each_df_Y[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11']].min(axis=1, skipna=True)
    each_df_Y.drop(['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8', 'Y9','Y10','Y11'], axis=1, inplace=True)
    df_user_Y_test.append(each_df_Y.copy())
    
    each_df_Z = each_df[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']]
    each_df_Z['ZMean'] = each_df_Z[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']].mean(axis=1, skipna=True)
    each_df_Z['ZStd'] = each_df_Z[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']].std(axis=1, skipna=True)
    each_df_Z['ZMax'] = each_df_Z[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']].max(axis=1, skipna=True)
    each_df_Z['ZMin'] = each_df_Z[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11']].min(axis=1, skipna=True)
    each_df_Z.drop(['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8', 'Z9','Z10','Z11'], axis=1, inplace=True)
    df_user_Z_test.append(each_df_Z.copy())
    
    # (1/n) * sqrt(x2 + y2 + z2)
    
    each_df_SR_0_s = (each_df['X0'].apply(np.square) + each_df['Y0'].apply(np.square) + each_df['Z0'].apply(np.square)).apply(np.sqrt)
    each_df_SR_1_s = (each_df['X1'].apply(np.square) + each_df['Y1'].apply(np.square) + each_df['Z1'].apply(np.square)).apply(np.sqrt)
    each_df_SR_2_s = (each_df['X2'].apply(np.square) + each_df['Y2'].apply(np.square) + each_df['Z2'].apply(np.square)).apply(np.sqrt)
    each_df_SR_3_s = (each_df['X3'].apply(np.square) + each_df['Y3'].apply(np.square) + each_df['Z3'].apply(np.square)).apply(np.sqrt)
    each_df_SR_4_s = (each_df['X4'].apply(np.square) + each_df['Y4'].apply(np.square) + each_df['Z4'].apply(np.square)).apply(np.sqrt)
    each_df_SR_5_s = (each_df['X5'].apply(np.square) + each_df['Y5'].apply(np.square) + each_df['Z5'].apply(np.square)).apply(np.sqrt)
    each_df_SR_6_s = (each_df['X6'].apply(np.square) + each_df['Y6'].apply(np.square) + each_df['Z6'].apply(np.square)).apply(np.sqrt)
    each_df_SR_7_s = (each_df['X7'].apply(np.square) + each_df['Y7'].apply(np.square) + each_df['Z7'].apply(np.square)).apply(np.sqrt)
    each_df_SR_8_s = (each_df['X8'].apply(np.square) + each_df['Y8'].apply(np.square) + each_df['Z8'].apply(np.square)).apply(np.sqrt)
    each_df_SR_9_s = (each_df['X9'].apply(np.square) + each_df['Y9'].apply(np.square) + each_df['Z9'].apply(np.square)).apply(np.sqrt)
    each_df_SR_10_s = (each_df['X10'].apply(np.square) + each_df['Y10'].apply(np.square) + each_df['Z10'].apply(np.square)).apply(np.sqrt)
    each_df_SR_11_s = (each_df['X11'].apply(np.square) + each_df['Y11'].apply(np.square) + each_df['Z11'].apply(np.square)).apply(np.sqrt)
    each_df_SR_frame = pd.DataFrame({0:each_df_SR_0_s, 1:each_df_SR_1_s, 2:each_df_SR_2_s, 3:each_df_SR_3_s, 4:each_df_SR_4_s,
                                    5:each_df_SR_5_s, 6:each_df_SR_6_s, 7:each_df_SR_7_s, 8:each_df_SR_8_s, 9:each_df_SR_9_s,
                                    10:each_df_SR_10_s, 11:each_df_SR_11_s})
    each_df_SR_frame['SR'] = each_df_SR_frame.mean(axis=1, skipna=True)
    each_df_SR_frame.drop([0,1,2,3,4,5,6,7,8,9,10,11], axis=1, inplace=True)
    df_user_square_root_test.append(each_df_SR_frame)
    
    # (1/n) cub(xyz)  mul

    f = lambda x: np.sign(x) * np.power(abs(x), 1./3)
    
    each_df_mul_0_s = (each_df['X0'] * each_df['Y0'] * each_df['Z0']).apply(f)
    each_df_mul_1_s = (each_df['X1'] * each_df['Y1'] * each_df['Z1']).apply(f)
    each_df_mul_2_s = (each_df['X2'] * each_df['Y2'] * each_df['Z2']).apply(f)
    each_df_mul_3_s = (each_df['X3'] * each_df['Y3'] * each_df['Z3']).apply(f)
    each_df_mul_4_s = (each_df['X4'] * each_df['Y4'] * each_df['Z4']).apply(f)
    each_df_mul_5_s = (each_df['X5'] * each_df['Y5'] * each_df['Z5']).apply(f)
    each_df_mul_6_s = (each_df['X6'] * each_df['Y6'] * each_df['Z6']).apply(f)
    each_df_mul_7_s = (each_df['X7'] * each_df['Y7'] * each_df['Z7']).apply(f)
    each_df_mul_8_s = (each_df['X8'] * each_df['Y8'] * each_df['Z8']).apply(f)
    each_df_mul_9_s = (each_df['X9'] * each_df['Y9'] * each_df['Z9']).apply(f)
    each_df_mul_10_s = (each_df['X10'] * each_df['Y10'] * each_df['Z10']).apply(f)
    each_df_mul_11_s = (each_df['X11'] * each_df['Y11'] * each_df['Z11']).apply(f)
    each_df_mul_frame = pd.DataFrame({0:each_df_mul_0_s, 1:each_df_mul_1_s, 2:each_df_mul_2_s, 3:each_df_mul_3_s, 4:each_df_mul_4_s,
                                    5:each_df_mul_5_s, 6:each_df_mul_6_s, 7:each_df_mul_7_s, 8:each_df_mul_8_s, 9:each_df_mul_9_s,
                                    10:each_df_mul_10_s, 11:each_df_mul_11_s})
    each_df_mul_frame['Mul'] = each_df_mul_frame.mean(axis=1, skipna=True)
    each_df_mul_frame.drop([0,1,2,3,4,5,6,7,8,9,10,11], axis=1, inplace=True)
    df_user_mul_test.append(each_df_mul_frame)
    
    # (1/n) sqrt(|xy + yz + xz|)  cross mul
    
    each_df_CM_0_s = (each_df['X0'] * each_df['Y0'] + each_df['Y0'] * each_df['Z0'] + each_df['X0'] * each_df['Z0']).apply(np.abs).apply(np.sqrt)
    each_df_CM_1_s = (each_df['X1'] * each_df['Y1'] + each_df['Y1'] * each_df['Z1'] + each_df['X1'] * each_df['Z1']).apply(np.abs).apply(np.sqrt)
    each_df_CM_2_s = (each_df['X2'] * each_df['Y2'] + each_df['Y2'] * each_df['Z2'] + each_df['X2'] * each_df['Z2']).apply(np.abs).apply(np.sqrt)
    each_df_CM_3_s = (each_df['X3'] * each_df['Y3'] + each_df['Y3'] * each_df['Z3'] + each_df['X3'] * each_df['Z3']).apply(np.abs).apply(np.sqrt)
    each_df_CM_4_s = (each_df['X4'] * each_df['Y4'] + each_df['Y4'] * each_df['Z4'] + each_df['X4'] * each_df['Z4']).apply(np.abs).apply(np.sqrt)
    each_df_CM_5_s = (each_df['X5'] * each_df['Y5'] + each_df['Y5'] * each_df['Z5'] + each_df['X5'] * each_df['Z5']).apply(np.abs).apply(np.sqrt)
    each_df_CM_6_s = (each_df['X6'] * each_df['Y6'] + each_df['Y6'] * each_df['Z6'] + each_df['X6'] * each_df['Z6']).apply(np.abs).apply(np.sqrt)
    each_df_CM_7_s = (each_df['X7'] * each_df['Y7'] + each_df['Y7'] * each_df['Z7'] + each_df['X7'] * each_df['Z7']).apply(np.abs).apply(np.sqrt)
    each_df_CM_8_s = (each_df['X8'] * each_df['Y8'] + each_df['Y8'] * each_df['Z8'] + each_df['X8'] * each_df['Z8']).apply(np.abs).apply(np.sqrt)
    each_df_CM_9_s = (each_df['X9'] * each_df['Y9'] + each_df['Y9'] * each_df['Z9'] + each_df['X9'] * each_df['Z9']).apply(np.abs).apply(np.sqrt)
    each_df_CM_10_s = (each_df['X10'] * each_df['Y10'] + each_df['Y10'] * each_df['Z10'] + each_df['X10'] * each_df['Z10']).apply(np.abs).apply(np.sqrt)
    each_df_CM_11_s = (each_df['X11'] * each_df['Y11'] + each_df['Y11'] * each_df['Z11'] + each_df['X11'] * each_df['Z11']).apply(np.abs).apply(np.sqrt)
    each_df_CM_frame = pd.DataFrame({0:each_df_CM_0_s, 1:each_df_CM_1_s, 2:each_df_CM_2_s, 3:each_df_CM_3_s, 4:each_df_CM_4_s,
                                    5:each_df_CM_5_s, 6:each_df_CM_6_s, 7:each_df_CM_7_s, 8:each_df_CM_8_s, 9:each_df_CM_9_s,
                                    10:each_df_CM_10_s, 11:each_df_CM_11_s})
    each_df_CM_frame['CM'] = each_df_CM_frame.mean(axis=1, skipna=True)
    each_df_CM_frame.drop([0,1,2,3,4,5,6,7,8,9,10,11], axis=1, inplace=True)
    df_user_cross_mul_test.append(each_df_CM_frame)
    
df_user_test_feature = []

for i in range(len(df_user_num_test)):
    tmp = pd.concat([df_user_test[i]['Class'], df_user_num_test[i], df_user_X_test[i], df_user_Y_test[i], df_user_Z_test[i], df_user_square_root_test[i], df_user_mul_test[i], df_user_cross_mul_test[i]],axis=1)
    df_user_test_feature.append(tmp.copy())

test_df = []
for each_subset in df_user_test_feature:
    if isinstance(test_df, list):
        test_df = each_subset
    else:
        test_df = pd.concat((test_df, each_subset), axis=0, ignore_index=True)
        
train_df = []
for each_subset in df_user_total_balanced:
    if isinstance(train_df, list):
        train_df = each_subset
    else:
        train_df = pd.concat((train_df, each_subset), axis=0, ignore_index=True)

# Finished train test split
normization = StandardScaler()
normization.fit(train_df.drop(['Class', 'Mul'], axis=1))
train_data = normization.transform(train_df.drop(['Class', 'Mul'], axis=1))
train_label = np.array(train_df['Class'])

test_data = normization.transform(test_df.drop(['Class', 'Mul'], axis=1))
test_label = np.array(test_df['Class'])

# Generating random data
random_input = np.random.randn(10000, 15)

###########################################################
# Bayes result
###########################################################
# Naive Bayes
gnb = GaussianNB()
gnb.fit(train_data, train_label) # Linear machine
print('Test Accuracy on Naive Bayes', gnb.score(test_data, test_label))
print('Train Accuracy on Naive Bayes', gnb.score(train_data, train_label))
confu_gnb = plot_confusion_matrix(gnb, test_data, test_label, normalize='true')
rand_pred = gnb.predict(random_input)
rand_class = np.zeros(5)
for each in rand_pred:
    rand_class[each - 1] += 1
print("Random output of NB is:", rand_class)
print('Distance from balance is:', np.sum(np.abs(rand_class - 2000)))

###########################################################
# SVM result
# final_pos[1]=5, final_pos[0]=1
# C=2.15443469, gamma=0.00464159
###########################################################
svmclf_best = svm.SVC(C=C[final_pos[1]], kernel='rbf', gamma=gamma[final_pos[0]])
svmclf_best.fit(train_data, train_label)
print('Test Accuracy on SVM', svmclf_best.score(test_data, test_label))
print('Train Accuracy on SVM', svmclf_best.score(train_data, train_label))
confu_svm = plot_confusion_matrix(svmclf_best, test_data, test_label, normalize='true')
rand_pred = svmclf_best.predict(random_input)
rand_class = np.zeros(5)
for each in rand_pred:
    rand_class[each - 1] += 1
print("Random output of SVM is:", rand_class)
print('Distance from balance is:', np.sum(np.abs(rand_class - 2000)))

###########################################################
# Perceptron result
###########################################################
perclf_best = Perceptron(eta0=per_final_eta)
perclf_best.fit(train_data, train_label)
print('Test Accuracy on Perceptron', perclf_best.score(test_data, test_label))
print('Train Accuracy on Perceptron', perclf_best.score(train_data, train_label))
confu_per = plot_confusion_matrix(perclf_best, test_data, test_label, normalize='true')
rand_pred = perclf_best.predict(random_input)
rand_class = np.zeros(5)
for each in rand_pred:
    rand_class[each - 1] += 1
print("Random output of Perceptron is:", rand_class)
print('Distance from balance is:', np.sum(np.abs(rand_class - 2000)))

###########################################################
# KNN result
###########################################################
knnclf_best = KNeighborsClassifier(n_neighbors=int(knn_final_nei))
knnclf_best.fit(train_data, train_label)
print('Test Accuracy on Perceptron', knnclf_best.score(test_data, test_label))
print('Train Accuracy on Perceptron', knnclf_best.score(train_data, train_label))
rand_pred = knnclf_best.predict(random_input)
rand_class = np.zeros(5)
for each in rand_pred:
    rand_class[each - 1] += 1
print("Random output of KNN is:", rand_class)
print('Distance from balance is:', np.sum(np.abs(rand_class - 2000)))

###########################################################
# Additional Perceptron (Phi Machine)
###########################################################
perclf_add = Perceptron(eta0=per_final_eta)
acc_padd = []
for k in range(len(df_user_total_balanced)):
    # balanced
    df_user_train = df_user_total_balanced[:k] + df_user_total_balanced[k+1:]
    df_user_val = df_user_total_balanced[k]
    # unbalanced
    # df_user_train = df_user_total[:k] + df_user_total[k+1:]
    # df_user_val = df_user_total[k]
    train_df = []
    val_df = df_user_val.copy()
    for each_subset in df_user_train:
        if isinstance(train_df, list):
            train_df = each_subset
        else:
            train_df = pd.concat((train_df, each_subset), axis=0, ignore_index=True)

    # Finished train test split
    normization.fit(train_df.drop(['Class', 'Mul'], axis=1))
    train_data = normization.transform(train_df.drop(['Class', 'Mul'], axis=1))
    train_label = np.array(train_df['Class'])

    val_data = normization.transform(val_df.drop(['Class', 'Mul'], axis=1))
    val_label = np.array(val_df['Class'])
    # Phi machine
    poly = PolynomialFeatures(2)
    train_data_poly = poly.fit_transform(train_data)
    val_data_poly = poly.fit_transform(val_data)
    
    perclf_add.fit(train_data_poly, train_label) # Linear machine
    acc_padd.append(perclf_add.score(val_data_poly, val_label))

print("Mean using Phi machine is", np.mean(acc_padd))
print("Std using Phi machine is", np.std(acc_padd))