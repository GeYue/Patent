
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

mms = MinMaxScaler(feature_range=(0, 1))

#x = np.array([[0.8370], [0.8297], [0.8238], [0.8200], [0.7985]])
x = np.array([[0.8370], [0.8297], [0.8238], [0.8200]])
print (x)

out = mms.fit_transform(x)
print (out)

#out2 = sigmoid(out)
#print (out2)

out3 = softmax(out)
print (out3)

#print (softmax(x))