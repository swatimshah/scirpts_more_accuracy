from sklearn import preprocessing
from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
from imblearn.over_sampling import SMOTE
import tensorflow
from sklearn.model_selection import train_test_split
from numpy import mean

MY_CONST = 40.
MY_CONST_NEG = -40.

def NormalizeData(data):
	print(numpy.amin(data))	
	print(numpy.amax(data))	
	return (data + (MY_CONST)) / (MY_CONST - (MY_CONST_NEG))

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

input = input.reshape(len(input), 4, 19)
input = input.transpose(0, 2, 1)

y_real = X[:, -1]

model = load_model('D:\\model_mychar20_9.h5')
y_pred = model.predict_proba(input) 
print(y_pred.shape)

#----------------------------------

y_corr = numpy.zeros((len(y_pred), 2))

for i in range(len(y_corr)):
	y_corr[i][1] = (y_pred[i][1] * 0.66)/((y_pred[i][1] * 0.66) + ((1 - y_pred[i][1]) * 1.32))
	y_corr[i][0] = ((1 - y_pred[i][1]) * 1.32)/((y_pred[i][1] * 0.66) + ((1 - y_pred[i][1]) * 1.32)) 

print(y_corr.shape)
print(y_corr)
#----------------------------------

y_max = numpy.argmax(y_pred, axis=1)
matrix = confusion_matrix(y_real, y_max)
print(matrix)
