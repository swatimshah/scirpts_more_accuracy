import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from pandas import DataFrame
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from numpy import savetxt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical 
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
import dill as pickle
from sklearn.pipeline import Pipeline
from numpy import asarray
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import learning_curve
from sklearn.ensemble import StackingClassifier
from keras.layers import Input
from keras.models import Model
from keras.losses import binary_crossentropy
from tensorflow.keras import regularizers
from numpy.random import seed
from tensorflow.random import set_seed
import tensorflow
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB



def NormalizeData(data):
	print(numpy.amin(data))	
	print(numpy.amax(data))	
	return (data + abs(numpy.amin(data))) / (numpy.amax(data) - numpy.amin(data))


# setting the seed
seed(1)
set_seed(1)

# load array
X_train_whole = loadtxt('d:\\tempflashes40.csv', delimiter=',')

# augment data
#choice = X_train_whole[:, -1] == 0.
#X_total = numpy.append(X_train_whole, X_train_whole[choice, :], axis=0)
#print(X_total.shape)

# balancing
#sm = SMOTE(random_state = 2)
#X_train_res, Y_train_res = sm.fit_resample(X_train_whole[:, 0:76], X_train_whole[:, -1].ravel())
#print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res == 1)))
#print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res == 0)))

tensorflow.compat.v1.reset_default_graph()
X_train, X_test, Y_train, Y_test = train_test_split(X_train_whole[:, 0:76], X_train_whole[:, -1], random_state=1, test_size=0.3, shuffle = True)
print(X_train.shape)
print(X_test.shape)


#=======================================

# Model configuration

input = numpy.zeros((0, 76))
testinput = numpy.zeros((0, 76))

MY_CONST = 40.
MY_CONST_NEG = -40.

mean_of_train = mean(X_train[:, 0:76])
print(mean_of_train)
input = X_train[:, 0:76] - mean_of_train
too_high_input = input > MY_CONST
input[too_high_input] = MY_CONST
too_low_input = input < MY_CONST_NEG
input[too_low_input] = MY_CONST_NEG
input = NormalizeData(input)
input_output = numpy.append(input, Y_train.reshape(len(Y_train), 1), axis=1) 
savetxt('d:\\input_output.csv', input_output, delimiter=',')

mean_of_test = mean(X_test[:, 0:76])
print(mean_of_test)
testinput = X_test[:, 0:76] - mean_of_test
too_high_testinput = testinput > MY_CONST
testinput[too_high_testinput] = MY_CONST
too_low_testinput = testinput < MY_CONST_NEG
testinput[too_low_testinput] = MY_CONST_NEG
testinput = NormalizeData(testinput)
savetxt('d:\\testinput.csv', testinput, delimiter=',')

#=====================================

print(len(input))
print(len(testinput))


# Create the model
#steps = [('svc', SVC(C=10.0, degree=5, kernel='poly', verbose=True, class_weight={0: 1, 1: 2}))]
#steps = [('svc', SVC(C=10.0, cache_size=200, class_weight={0: 2, 1: 1}, coef0=0.0,
#  decision_function_shape='ovr', degree=4, gamma='auto', kernel='poly',
#  max_iter=5000, probability=True, random_state=1, shrinking=True,
#  tol=0.0001, verbose=True
#))]

#steps = [('lda', LinearDiscriminantAnalysis(n_components=1, tol=0.001, solver='svd', priors=[0.66, 0.34]))]
steps = [('mlp', MLPClassifier(hidden_layer_sizes=(100, 30), random_state=1, max_iter=1200, verbose=True, learning_rate_init=0.001, solver='adam', alpha=0.001, batch_size=32))]

model = Pipeline(steps=steps)
model.fit(input, Y_train)

# evaluate the model
Y_hat_classes = model.predict(testinput)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)

#==================================

filename = 'LDA_NB_model_small_matrix_9.sav'
pickle.dump(model, open(filename, 'wb'))

#==================================

#Removed dropout and reduced momentum and reduced learning rate