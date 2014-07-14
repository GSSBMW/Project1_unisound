
import numpy as np
import svd

def correctnessVerification(levelMatFileName, inputMatFileName, outputFileName):
    """Show the variation trend of distance between original 
    output and output after dimention reduction, with the 
    change of k. And write this trend in to file.
    """
    levelMatFile = open(levelMatFileName,'r')
    inputMatFile = open(inputMatFileName,'r')
    outputFile = open(outputFileName,'w')
    
    #read level matrix from levelMatFile, named levelMat
    while True:
   	l = levelMatFile.readline()
	if len(l)==0 :
            break
   	l = l.translate(None, '\n[]<>').strip()
   	if len(l)!=0 :
   	    l = l.split(' ')
   	    if l[0]=='affinetransform':
   		inmatrix = True;
   		levelMat = []
   		continue
   	    elif l[0]=='sigmoid' or l[0]=='softmax':
   		inmatrix = False
   
                # process constant vector b
   	        for i in range(len(levelMat)-1):
   		    levelMat[i].insert(0,levelMat[len(levelMat)-1][i])
   		del levelMat[len(levelMat)-1]
                levelMat = np.array(levelMat)
   				
                break
   	    if inmatrix :
                   levelMat.append([float(member) for member in l])
    #read input matrix form inputMatFile, named inputMat
    inputMat = []
    while True:
        l = inputMatFile.readline().translate(None, '\n[]').strip()
        if len(l):
            l = l.split(' ')
            l.insert(0,'1.0')
            inputMat.append([float(member) for member in l])
        else:
            break
    inputMat = np.array(inputMat)
    outputMat = np.dot(levelMat, inputMat.T)

    #change the threshold
    U, Sigma, Vt = np.linalg.svd(levelMat)
    reducedDim = svd.getDim(Sigma, 1.0)
    if reducedDim:
        U, Sigma, Vt = svd.dimReduce(U, Sigma, Vt, reducedDim)
        levelMat_reduce = np.dot(U, np.dot(np.diag(Sigma),Vt) )
        outputMat_reduce = np.dot(levelMat_reduce, inputMat.T) 
    else:
        print 'There is an error!\n'
        return
    
    distanceVector = np.sqrt(np.sum(np.square(outputMat_reduce-outputMat), axis=0))
    distance = np.sum(distanceVector)/distanceVector.shape[0]
    print 'Distance vector: ',distanceVector
    print 'Distance:',distance


if __name__=='__main__':
    correctnessVarification('mydata_level',\
            'inputVector_2x6.data','testDistance.data')
