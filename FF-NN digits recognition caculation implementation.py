#Snir Koska
#To reduce the use of loops in the calculation, the training process is carried out in matrix form across all  
#data samples

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import softmax
from scipy.special import expit

#Load MNIST data
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

start_time = time.time()

def forward_pass(data, w, net_size, labels_one_hot):
    
    a = {}
    z = {}
    data_size = np.shape(data)[0] # number of data examples
    
    a[str(0)] = data
    z[str(0)] = data
    for i in range(len(net_size)-1):

        a[str(i+1)] = np.dot(z[str(i)], w[str(i)].T) #hidden neurons value before activation using dot product
        z[str(i+1)] = expit(a[str(i+1)]/np.sqrt(data_size))     #hidden neurons value after sigmoid activation
        z[str(i+1)][:,net_size[i+1]-1] = 1    #bias is implemented without the b terms, but with added weights connected
                                            #to added "1 nodes" in the input and hidden layer, this line forces these nodes to stay with value 1
        if i == len(net_size)-2:
            z[str(i+1)] = softmax(a[str(i+1)], axis =1) #output neuron value after a softmax activation softmax along rows axis
        

    E=(1/data_size)*np.sum(np.power((z[str(len(net_size)-1)]-labels_one_hot),2)) #calculation of mean square error
    
    return E, z

def backprop(z, w, net_size, labels_one_hot):
    
    error = {}
    error[str(len(net_size)-1)] = z[str(len(net_size)-1)]*(1-z[str(len(net_size)-1)])*(z[str(len(net_size)-1)]-labels_one_hot) #calculation of output error (delta) term for output layer 

    for i in reversed(range(len(net_size)-1)):
        error[str(i)] = z[str(i)]*(1-z[str(i)])*(np.dot(error[str(i+1)], w[str(i)])) #calculation of inner layers error

    return error

def train (data, labels, X_test, t_test, net_size, eta, iter_num): #main code loop - training process

    labels_one_hot = np.eye(10)[list(map(int,np.ndarray.tolist(labels)))] #transfer train labels into 1-hot vector format sized 10
    test_one_hot = np.eye(10)[list(map(int,np.ndarray.tolist(t_test)))]
    
    E_array_train = np.zeros(iter_num) #array to save MSE calculations
    w = {}
    acc_array_train=np.zeros(iter_num)
    acc_array_test=np.zeros(iter_num)
    data_size = np.shape(data)[0] # number of data examples

    for i in range(len(net_size)-1):
        w[str(i)] = np.random.standard_normal((net_size[i+1],net_size[i])) #initialization of all weights (normal distribution)
                   

    for i in range(iter_num):
        
        E, z = forward_pass(data, w, net_size, labels_one_hot)
        error = backprop(z, w, net_size, labels_one_hot)
        
        E_test, z_test = forward_pass(X_test, w, net_size, test_one_hot)
        
        for j in range(len(net_size)-1):

            w[str(j)] = w[str(j)] - (eta*(1/data_size)*(np.dot(z[str(j)].T, error[str(j+1)])).T) #update step for hidden weights (summing across all training samples)
        
            E_array_train[i] = E #save MSE data
            
        train_pred_label = np.argmax(z[str(len(net_size)-1)], axis=1) #prediction label vector for each example based on argmax of probability along coloumns axis
        acc_train = np.sum((train_pred_label == labels.astype('int64')))/np.size(labels) #compute accuracy on train data
        acc_array_train[i] = acc_train
        
        
        test_pred_label = np.argmax(z_test[str(len(net_size)-1)], axis=1) #prediction label vector for each test example based on argmax of probability along coloumns axis
        acc_test = np.sum((test_pred_label == t_test.astype('int64')))/np.size(t_test) #compute accuracy on test data
        acc_array_test[i] = acc_test
        
    return E_array_train, z, error, w, acc_array_train, acc_array_test

net_size = np.array([785, 10]) # each number in this array represent the # of neurons in the spesific layer
                                        # from input to output (+1 for bias except in output layer)
eta = 7 #learning rate
iter_num = 100 # number of iterations to go over full batch in training process         


X, t = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

#Shuffle data before dividing into groups
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
t = t[permutation]
X = X.reshape((X.shape[0], -1))

X=np.append(X,np.ones((np.shape(X)[0],1)),axis=1) #add 1's coloumn at end of  data

X=X.astype(np.longdouble) #Change examples datatype to float128 as to prevent overflow in calculations of exponents and divisions, will take longer to calculate.
t=t.astype('int64') # Change the labels into array of int64 for simplicity of calculations.

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2) #Split data into train and test

# The next lines standardize the images
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train[0:5000,:]
t_train = t_train[0:5000]

E_array, z, error, w, acc_array_train, acc_array_test = train(X_train, t_train, X_test, t_test, net_size, eta, iter_num) 

figure, axis = plt.subplots(1,3) 
   
axis[0].plot(E_array)
axis[0].set_title("Mean square error")
axis[0].set_xlabel("#Iteration")   

axis[1].plot(acc_array_train)
axis[1].set_title("Train accuracy")
axis[1].set_xlabel("#Iteration") 

axis[2].plot(acc_array_test)
axis[2].set_title("Test accuracy")
axis[2].set_xlabel("#Iteration")  
 
plt.show()
 
print("--- %s seconds ---" % (time.time() - start_time))