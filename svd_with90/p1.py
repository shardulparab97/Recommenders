#main program for findin the SVD Values
import pandas as pd
import numpy as np
import numpy
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy.sparse.linalg import eigsh
from scipy import sparse
import math
import pickle
import time

data_shape = (943, 1683)

def find_svd(A,k):
    At=csr_matrix(A.transpose())
    AAt=A*At
    AtA=At*A
    print "Calculating EigenVectors for AAt"

    #calculating U
    eigenvalue,U=eigsh(AAt,k)
    idx = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[idx]
    U = U[:,idx]

    #eigenvalue is a vector containg k eigenvalues
    eigenvalue,V=eigsh(AtA,k)
    #print (eigenvalue)
    #print (V)
    idx = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[idx]
    V = V[:,idx]
    #print ("Eigen Value is :",eigenvalue)
    #print ("idx=",idx)
    #print ("V is",V)
    Vt=V.transpose()
    nonzeroeig= filter(lambda a: a > 0.001, eigenvalue)
    x=len(nonzeroeig)
    sigma = np.zeros(shape=(k,k))
    for i in range(0,x):
	    sigma[i][i]=math.sqrt(nonzeroeig[i])
    sigmasparse=sparse.csr_matrix(sigma)
    return U,sigma,Vt

#X_train = scipy.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=numpy.float, shape=data_shape)
df = pd.read_csv( "../data/ua.base", sep="\t", header=-1)
values = df.values
values[:, 0:2] -= 1 #making all values of the first and second columns 0 indexed by subtractng -1

#print values
X_train=scipy.sparse.csr_matrix((values[:,2],(values[:,0],values[:,1])),dtype=np.float, shape = data_shape)


df = pd.read_csv("../data/ua.test", sep="\t", header=-1)
values = df.values
values[:, 0:2] -= 1
X_test = scipy.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=numpy.float, shape=data_shape)

X_row_mean = numpy.zeros(data_shape[0])
X_row_sum = numpy.zeros(data_shape[0])

train_rows, train_cols = X_train.nonzero() #getting all non zero values of rows and columns

# Iterate through nonzero elements to compute sums and counts of rows elements
for i in range(train_rows.shape[0]):
    X_row_mean[train_rows[i]] += X_train[train_rows[i], train_cols[i]]
    X_row_sum[train_rows[i]] += 1

# Note that (X_row_sum == 0) is required to prevent divide by zero
X_row_mean /= X_row_sum + (X_row_sum == 0)

# Subtract mean rating for each user
for i in range(train_rows.shape[0]):
    X_train[train_rows[i], train_cols[i]] -= X_row_mean[train_rows[i]]


#Same for test rows too!
test_rows, test_cols = X_test.nonzero()
for i in range(test_rows.shape[0]):
    X_test[test_rows[i], test_cols[i]] -= X_row_mean[test_rows[i]]

X_train = numpy.array(X_train.toarray())
X_test = numpy.array(X_test.toarray())

t1=time.clock()
U,sigma,Vt=find_svd(X_train,100)
t2=time.clock()
print ("U")
print (U.shape)
print ("sigma")
print (sigma.shape)
print ("Vt")
print (Vt.shape)

X_pred = U.dot(sigma)
X_pred= X_pred.dot(Vt)

#Again adding the mean
for i in range(X_row_mean.shape[0]):
    for j in range(X_pred.shape[1]):
        X_pred[i][j]+=X_row_mean[i]


print (X_pred)
f=open('X_pred.p','w')
pickle.dump(X_pred, f)
f.close()

f=open('U.p','w')
pickle.dump(U, f)
f.close()

f=open('sigma.p','w')
pickle.dump(sigma, f)
f.close()

f=open('Vt.p','w')
pickle.dump(Vt, f)
f.close()

f=open('X_row_mean.p','w')
pickle.dump(X_row_mean, f)
f.close()

#Getting the total energy first
total_energy=0
sigma_list=[]
for i in range(sigma.shape[0]):
    for j in range(sigma.shape[1]):
        if(sigma[i][j]>0):
            #print(sigma[i][j])
            total_energy+=sigma[i][j]**2
            sigma_list.append(sigma[i][j])


print ("Total Energy =",total_energy)

#finding total energy 2
total_energy2=0
count=-1

for item in sigma_list:
    count+=1
    total_energy2+=item**2
    #avoid exceeding over 90
    if(total_energy2/total_energy>0.9):
        break

print ("Ratio:",total_energy2/total_energy)
print ("Count values taken =",count)
c=count

count=-1
t3=time.clock()
for i in range(sigma.shape[0]):
    for j in range(sigma.shape[1]):
        if(sigma[i][j]>0):
            count+=1
            if(count>c):
                sigma[i][j]=0
t4=time.clock()


X_pred_new=U.dot(sigma)
X_pred_new=X_pred_new.dot(Vt)


f=open('X_pred_new.p','w')
pickle.dump(X_pred_new, f)
f.close()


