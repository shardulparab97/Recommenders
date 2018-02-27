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
from scipy.sparse import csr_matrix,hstack,vstack
from scipy.sparse import linalg
from scipy.sparse.linalg import eigsh
from scipy.sparse import lil_matrix
import math
import random
import time

data_shape = (943, 1683)


d = pd.read_csv( "../data/ua.base", sep="\t", names =["userId","movieId","rating","timestamp"])
values = d.values
values[:, 0:2] -= 1 #making all values of the first and second columns 0 indexed by subtractng -1


userIdl = list(d.userId.unique())
movieIdl = list(d.movieId.unique())
data = d['rating'].tolist()
m=len(userIdl)
n=len(movieIdl)
row = d.userId.astype('category', categories=userIdl).cat.codes #row contains all the userIds along with their serialId
col = d.movieId.astype('category', categories=movieIdl).cat.codes #column contains all the moviesIds along woth the serialId
A = csr_matrix((data, (row, col)), shape=(m,n),dtype=np.float64).tolil()
square=0
rowSqrArr=[]
colSqrArr=[]

A_row_mean = np.zeros(data_shape[0])
A_row_sum = np.zeros(data_shape[0])

train_rows, train_cols = A.nonzero()

for i in range(train_rows.shape[0]):
    A_row_mean[train_rows[i]] += A[train_rows[i], train_cols[i]]
    A_row_sum[train_rows[i]] += 1

# Note that (X_row_sum == 0) is required to prevent divide by zero
A_row_mean /= A_row_sum + (A_row_sum == 0)

# Subtract mean rating for each user
for i in range(train_rows.shape[0]):
   	A[train_rows[i], train_cols[i]] -= A_row_mean[train_rows[i]]

#print A

print (time.clock())
print ("Calculating total sum")
#calculating square sum of all ratings
for i in range(0,m):
	for j in range(0,n):
		if A[i,j]!=0:
			square+=A[i,j]*A[i,j]
print (time.clock())
print ("Calculating row probabilities")
#calculating row probablities
rowsquare=0
for i in range(0,m):
	for j in range(0,n):
		if A[i,j]!=0:
			rowsquare+=A[i,j]*A[i,j]
	rowSqrArr.append(rowsquare/square)
print (time.clock())
print ("Calculating column probabilities")
#calculating column probablities
colsquare=0
for i in range(0,n):
	for j in range(0,m):
		if A[j,i]!=0:
			colsquare+=A[j,i]*A[j,i]
	colSqrArr.append(colsquare/square)
#column select
selectCol=[]
print (time.clock())
c=input("Enter no of columns you want to select: ")
print ("selecting columns")
C=lil_matrix((m,0))
for i in range(0,c):
	rand=random.random()
	for j in range(0,n):
		if(rand<colSqrArr[j]):
			selectCol.append(j)
			C=hstack([C,A.getcol(j)]) #combining it to add column
			break

print (time.clock())
#row select
selectRow=[]
r=input("Enter no of rows you want to select: ")
print ("selecting rows")
R=lil_matrix((0,n))
for i in range(0,r):
	rand=random.random()
	for j in range(0,m):
		if(rand<rowSqrArr[j]):
			selectRow.append(j)
			R=vstack([R,A.getrow(j)]) #add row
			break

print (time.clock())

print ("Finding intersection")
inter=[]
for i in range(0,r):
	a=selectRow[i]
	row=[]
	for j in range(0,c):
		b=selectCol[j]
		row.append(A[a,b])
	inter.append(row)
print (time.clock())
print ("Calculating SVD")

W = lil_matrix(inter)
Wt=lil_matrix(W.transpose())
WWt=W*Wt
WtW=Wt*W
eigenvalue,X=eigsh(WWt,r-1)
idx = eigenvalue.argsort()[::-1]
eigenvalue = eigenvalue[idx]
X = X[:,idx]
eigenvalue2,Y=eigsh(WtW,c-1)
idx2 = eigenvalue2.argsort()[::-1]
eigenvalue2 = eigenvalue2[idx2]
Y = Y[:,idx2]
Yt=Y.transpose()
nonzeroeig= filter(lambda a: a > 0.001, eigenvalue2)
l=len(nonzeroeig)
#calculating preudo inverse
pseudosigma = np.zeros(shape=(c-1,r-1))
for i in range(0,l):
	pseudosigma[i][i]=(1/math.sqrt(abs(nonzeroeig[i])))**2
Xt=X.transpose()
U=np.dot(Y,pseudosigma)
U=np.dot(U,Xt)

print "C:-"
print C
print "U:-"
print U
print "R:-"
print R
print time.clock()
A_pred=C*U
A_pred=A_pred*R
for i in range(0,m):
	for j in range(0,n):
		if(abs(A_pred[i,j])<0.001):
			A_pred[i,j]=0
print "A:-"
print A_pred
print time.clock()

for i in range(A_row_mean.shape[0]):
    for j in range(A_pred.shape[1]):
        A_pred[i][j]+=A_row_mean[i]

f=open('./C.p','w')
pickle.dump(C, f)
f.close()

f=open('./U.p','w')
pickle.dump(U, f)
f.close()

f=open('./R.p','w')
pickle.dump(R, f)
f.close()

f=open('./A_pred.p','w')
pickle.dump(A, f)
f.close()



