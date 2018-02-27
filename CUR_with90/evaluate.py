import math
import pickle
import pickle
import math
import pandas as pd
import numpy as np
import scipy
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
from scipy.linalg import norm
import math

with open('C.p', 'rb') as f:
    c = pickle.load(f)


with open('U.p', 'rb') as f:
    u = pickle.load(f)


with open('R.p', 'rb') as f:
    r = pickle.load(f)


with open('A_pred.p', 'rb') as f:
    A_pred = pickle.load(f)


def loadTestSet():

    df = pd.read_csv("../data/ua.test", sep="\t", header=-1)
    values = df.values
    values[:, 0:2]-= 1

    X_test=scipy.sparse.csr_matrix((values[:,2],(values[:,0],values[:,1])),dtype=np.float, shape = (943,1680))

    X_test=X_test.toarray()

    return X_test


def rmse(trainData,testData):
    val=0
    count=0
    for i in range(testData.shape[0]):
        for j in range(testData.shape[1]):
            if trainData[i][j]>0:
                val+=(testData[i][j] - trainData[i][j]) ** 2
                count+=1
            if(trainData[i][j]>5):
                val+=(testData[i][j] - 5) ** 2
                count+=1
            if(trainData[i][j]<0):
                val+=(testData[i][j]) ** 2
                count+=1
    return math.sqrt(val/count)

def findPrecisionOnTopk(trainData,testData):
    num=0
    k=0
    den=0
    total_ranking_list = []
    for i in range(testData.shape[0]):
        for j in range(testData.shape[1]):
            if testData[i][j]>=3.5:
                k+=1
                if (trainData[i][j]>=3.5):
                    num+=1

    for i in range(trainData.shape[0]):
        for j in range(trainData.shape[0]):
            total_ranking_list.append(trainData[i][j])

    total_ranking_list.sort()
    total_ranking_list.reverse()
    for val in total_ranking_list[0:k]:
        if val>=3.5:
            den+=1

    return float(num)/float(den) * 100


def spearmans_coefficient(trainData,testData):
    set_1=[]
    set_2=[]
    for i in range(testData.shape[0]):
        for j in range(testData.shape[1]):
            if testData[i][j]>0:
                    set_2.append(testData[i][j])
                    set_1.append(trainData[i][j])


    set_1_ord = sorted(set_1)
    set_2_ord = sorted(set_2)

    set_1_ranked = []
    set_2_ranked = []

    for i in range(len(set_1)):
	    set_1_ranked.append([set_1[i], set_1_ord.index(set_1[i])+1])

    for i in range(len(set_2)):
	    set_2_ranked.append([set_2[i], set_2_ord.index(set_2[i])+1])

    d = []
    for i in range(len(set_1_ranked)):
	    d.append(set_1_ranked[i][1] - set_2_ranked[i][1])


# calculate d^2
    d_sq = [i**2 for i in d]


# sum d^2
    sum_d_sq = sum(d_sq)

# calculate n^3 - n
    n_cu_min_n = len(set_1)**3 - len(set_1)

    rho = 1- ((6.0*sum_d_sq)/n_cu_min_n)
    return 1+rho

testData = loadTestSet()
#print (calculateRMSE(A_pred,testData))
def top_k_precision(pred, test,factor=0.002):
	'''
	Calculate Precision@top k.
	Inputs:
	pred (1D numpy array): numpy array containing predicted values.
	test (1D numpy array): numpy array containing the ground truth values.

	Returns:
	(float): average Precision@top k.
	'''
	# THRESHOLD=3.5
	# K=5

	precision_list=[]
        precision_list=[]
        for i in range(test.shape[0]):
            temp_df={}
            no_equals=0
            count=0
            for j in range(test.shape[1]):
                temp_df['rating']=test[i][j]>3.5
                if(test[i][j]>3.5):
                    count+=1
                temp_df['prediction']=pred[i][j]>=3.5
                no_equals+=temp_df['rating']==temp_df['prediction']
            if(count==0):
                temp_precision=0
                continue
            temp_precision=no_equals/float(count)
            precision_list.append((temp_precision))


	return np.mean(np.array(precision_list))*factor

def spearman_rank_correlation(pred, truth):
    d=0
    for i in range(truth.shape[0]):
        for j in range(truth.shape[1]):
            if(truth[i][j]>0):
                d+=math.pow((pred[i][j]-truth[i][j]),2)



	n=len(pred)
	rho=1-6.0*d/(n*(n*n-1))
	return rho


A_pred=A_pred.toarray()
A_pred=np.array(A_pred)

print ("RMSE VALUES ARE:",rmse(A_pred,testData))
print ("Precision on top k:",top_k_precision(A_pred,testData))
print ("Spearman s correlation coefficient:",spearman_rank_correlation(A_pred,testData))
