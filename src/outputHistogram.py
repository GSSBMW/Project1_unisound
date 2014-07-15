import numpy as np
import matplotlib.pyplot as plt
import svd
from correctnessVerification import readLevelMat
from correctnessVerification import readInputMat

def outputHistogram(levelMatFileName, inputMatFileName, k=0.99):
    """Plot output historgram of sublevel, comming from one original level. 
    And compare differences between original and reduced mode.
    """
    k_str = raw_input("Please enter value of k (0.0~1.0 or 'q'): ")
    if k_str=='q':
        pass
    else:
        k = float(k_str)

    levelMatFile = open(levelMatFileName,'r')
    inputMatFile = open(inputMatFileName,'r')
 
    levelMat = readLevelMat(levelMatFile)
    inputMat = readInputMat(inputMatFile)
    U, Sigma, Vt = np.linalg.svd(levelMat)

    #original
    U_org, Sigma_org, Vt_org = svd.dimReduce(U, Sigma, Vt, len(Sigma))
    subLevelMat1_org = np.dot(np.diag(Sigma_org), Vt_org)
    subLevelMat2_org = U_org
    subLevelOutputMat1_org = np.dot(subLevelMat1_org, inputMat.T)
    subLevelOutputMat2_org = np.dot(subLevelMat2_org, subLevelOutputMat1_org)

    #reduced
    reducedDim = svd.getDim(Sigma, k)
    if reducedDim:
        U_r, Sigma_r, Vt_r = svd.dimReduce(U, Sigma, Vt, reducedDim)
    else:
            print 'There is an error!\n'
            return
    subLevelMat1_r = np.dot(np.diag(Sigma_r),Vt_r)
    subLevelMat2_r = U_r
    subLevelOutputMat1_r = np.dot(subLevelMat1_r, inputMat.T)
    subLevelOutputMat2_r = np.dot(subLevelMat2_r, subLevelOutputMat1_r)

    #plot
    fig1 = plt.figure('Output histogram of sublevel 1')
    plt.subplot(2,1,1)
    plt.hist(subLevelOutputMat1_org.reshape(
        (subLevelOutputMat1_org.shape[0]*subLevelOutputMat1_org.shape[1],)),
        100,(-150,150))
    plt.xlabel('output_value'); plt.ylabel('amount')
    plt.title('Original dimention (%d)'%len(Sigma))
    plt.xlim(-150,150)
    plt.subplot(2,1,2)
    plt.hist(subLevelOutputMat1_r.reshape(
        (subLevelOutputMat1_r.shape[0]*subLevelOutputMat1_r.shape[1],)),
        100,(-150,150))
    plt.xlabel('output_value'); plt.ylabel('amount')
    plt.title('Reduced dimention (%d)'%reducedDim)
    plt.xlim(-150,150)

    fig2 = plt.figure('Output histogram of sublevel 2')
    plt.subplot(2,1,1)
    plt.hist(subLevelOutputMat2_org.reshape(
        (subLevelOutputMat2_org.shape[0]*subLevelOutputMat2_org.shape[1],)),
        100,(-150,150))
    plt.xlabel('output_value'); plt.ylabel('amount')
    plt.title('Original dimention (%d)'%len(Sigma))
    plt.xlim(-150,150)
    plt.subplot(2,1,2)
    plt.hist(subLevelOutputMat2_r.reshape(
        (subLevelOutputMat2_r.shape[0]*subLevelOutputMat2_r.shape[1],)),
        100,(-150,150))
    plt.xlabel('output_value'); plt.ylabel('amount')
    plt.title('Reduced dimention (%d)'%reducedDim)
    plt.xlim(-150,150)

    plt.show()
if __name__=='__main__':
    outputHistogram('../data/dnn_l3.data','../data/inputVector_1000x512.data')
