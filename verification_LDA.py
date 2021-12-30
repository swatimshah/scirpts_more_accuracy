from sklearn import preprocessing
from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import keras.backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope
from numpy import mean
import pickle


MY_CONST = 40.
MY_CONST_NEG = -40.

def NormalizeData(data):
    return (data + (MY_CONST)) / (MY_CONST - (MY_CONST_NEG))

model_file = open ("LDA_NB_model_small_matrix_9.sav", "rb")
model = pickle.load(model_file)

X = loadtxt('d:\\flashes_1_to_12_152_small_mat_testing-B-1-to-12.csv', delimiter=',')

mean_of_test = mean(X[:, 0:76])
print(mean_of_test)
input = X[:, 0:76] - mean_of_test
too_high_input = input > MY_CONST
input[too_high_input] = MY_CONST
too_low_input = input < MY_CONST_NEG
input[too_low_input] = MY_CONST_NEG
input = NormalizeData(input)
savetxt('d:\\input-swati-online.csv', input, delimiter=',')


y_real = X[:, -1]

y_pred = model.predict(input) 


#----------------------------------

#y_corr = numpy.zeros((len(y_pred), 2))

#for i in range(len(y_corr)):
#	y_corr[i][1] = (y_pred[i][1] * 0.4)/((y_pred[i][1] * 0.4) + ((1 - y_pred[i][1]) * 1.6))
#	y_corr[i][0] = ((1 - y_pred[i][1]) * 1.6)/((y_pred[i][1] * 0.4) + ((1 - y_pred[i][1]) * 1.6)) 

#print(y_corr.shape)
#print(y_corr)
#----------------------------------

#y_max = numpy.argmax(y_pred, axis=1)
matrix = confusion_matrix(y_real, y_pred)
print(matrix)




