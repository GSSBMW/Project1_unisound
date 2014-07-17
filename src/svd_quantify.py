# Singular Value Decomposition and quantification

import numpy as np

def getDim(Sigma, threshold):
	"""Reduce the dimention of Sigma according to the treshold of Energy."""
	if not((threshold>0.0) and (threshold<=1.0)):
		print "\tError: Threshold is illegal.\n"
		return 0

	#aim at imprecision of float type in python
	if threshold==1.0:
		return len(Sigma)

	sumEnergy = 0.0;
	for i in range(len(Sigma)):
		sumEnergy += Sigma[i]**2

	for i in range(len(Sigma)):
		if i==0 :
			curEnergy = Sigma[0]**2
			if curEnergy/sumEnergy >= threshold :
				break
			continue
		curEnergy += Sigma[i]**2
		if curEnergy/sumEnergy >= threshold :
			break
	
	return (i+1)


def dimReduce(U, Sigma, Vt, dim):
	"""Reduce the dimention of U, Sigma and Vt to specified dim in one axis"""
	return (U[:,0:dim],Sigma[0:dim],Vt[0:dim,:])

def rowQuantify(matrix, r=127.):
    """Quantify every row of matrix in range -r ~ +r.
    Return matrix after quantified and quantification vector.
    """
    quantifyVector = []
    for i in range(matrix.shape[0]):
        m = np.abs(matrix[i,:]).max()
        if m:
            matrix[i,:] = np.around(matrix[i,:]*r/m)
            quantifyVector.append(m/r)
        else:
            quanrifyVector.append(1.0)
    return matrix, quantifyVector

def levelMat2File_quantify(levelMatrix, quantifyVector, fileName, firstColAsCon=True, activeStr='sigmoid'):
	"""Append parameter matrix and quantification parameter of one level 
        to a file after quantified.
	firstColAsCon is True means that get the first column of levelMatrix 
        as constant vector b. Otherwise, vector b is 0.
	Value of activeStr is 'sigmoid' or 'softmax.'
	"""
	f = open(fileName,'a')

	m,n = levelMatrix.shape
	if firstColAsCon:
	    startCol=1
	    f.write('<affinetransform> %d %d\n [\n'%(m,n-1))
	else:
	    startCol=0
	    f.write('<affinetransform> %d %d\n [\n'%(m,n))
	for i in range(m):
	    f.write('  ')
	    for j in range(startCol, n):
	        f.write('%.0f '%levelMatrix[i][j])
            if i==(m-1):
    		f.write(']\n')
	    else:
	    	f.write('\n')
	f.write(' [ ')
	for i in range(m):
            if firstColAsCon:
		f.write('%.0f '%levelMatrix[i][0])
	    else:
		f.write('0 ')
	f.write(']\n<quantify> [ ')
        for i in range(len(quantifyVector)):
            f.write('%.10f '%quantifyVector[i])
	f.write(']\n<%s> %d %d \n'%(activeStr,m,m))
	
	f.close()


def levelMat2File(levelMatrix, fileName, firstColAsCon=True, activeStr='sigmoid'):
	"""Append parameter matrix of one level to a file.
	firstColAsCon is True means that get the first column 
	of levelMatrix as constant vector b. Else, vector b is 0.
	Value of activeStr is 'sigmoid' or 'softmax.'
	"""
	f = open(fileName,'a')

	m,n = levelMatrix.shape
	if firstColAsCon:
		startCol=1
		f.write('<affinetransform> %d %d\n [\n'%(m,n-1))
	else:
		startCol=0
		f.write('<affinetransform> %d %d\n [\n'%(m,n))
	for i in range(m):
		f.write('  ')
		for j in range(startCol, n):
			f.write(str(levelMatrix[i][j])+' ')
		if i==(m-1):
			f.write(']\n')
		else:
			f.write('\n')
	f.write(' [ ')
	for i in range(m):
		if firstColAsCon:
			f.write(str(levelMatrix[i][0])+' ')
		else:
			f.write('0 ')
	f.write(']\n')
	f.write('<%s> %d %d \n'%(activeStr,m,m))
	
	f.close()

def singularValueDec(sourceFile, destinationFile, threshold=0.99, k=0):
	"""Make singular value decompostion of all matrix in sourcefile,
	and write new marix of parameters to destinationFile in specified
	format after quantified. 
	If k is specified, value of reduced dimention is k. Otherwise, reduced
	dimention is determined by threshold.
        File format of each level:      File format of quantifiaed level:
	<affinetranform> 2 3            <affinetransform> 2 3
	 [                              [
	  1 2 3                          42 84 127
	  4 5 6 ]                        85 106 127 ]
	 [ 1 1 ]                        [ 42 21 ]
	<sigmoid> 2 2                   <quantify> [ 0.0236220472 0.0472440945]
                                        <sigmoid> 2 2
	"""
	f = open(sourceFile,'r')
	tmpf = open(destinationFile, 'w')
	tmpf.close()
        isFirstLevel=True
	while True:
		l = f.readline()
		if len(l)==0 :
			break
		l = l.translate(None, '\n[]<>').strip()
		if len(l)!=0 :
			l = l.split(' ')
			if l[0]=='affinetransform':
				inmatrix = True;
				matrix_A = []
				continue
			elif l[0]=='sigmoid' or l[0]=='softmax':
				inmatrix = False

                                # process constant vector b
				for i in range(len(matrix_A)-1):
					matrix_A[i].insert(0,matrix_A[len(matrix_A)-1][i])
				del matrix_A[len(matrix_A)-1]
				
                                matrix_A = np.array(matrix_A)
				
				#process matrix
				U, Sigma, Vt = np.linalg.svd(matrix_A)

				if k:
					reducedDim = k;
				else:
					reducedDim = getDim(Sigma, threshold)
				
				if reducedDim:
					U,Sigma,Vt = dimReduce(U, Sigma, Vt, reducedDim)
                                
                                #show Vt[0,0] is much larger than other value
                                #print 'Vt:  ',Vt[0:10,0:10],'\n\n'
                                
				#write matrix to file
                                if isFirstLevel:
		    		    levelMat2File(np.dot(np.diag(Sigma),Vt),destinationFile)
				    levelMat2File(U,destinationFile,False,l[0])
                                    isFirstLevel=False
                                else:
                                    subLevelMat1,qv1 = rowQuantify(np.dot(np.diag(Sigma),Vt))
                                    subLevelMat2,qv2 = rowQuantify(U)
				    levelMat2File_quantify(subLevelMat1,qv1,destinationFile)
				    levelMat2File_quantify(subLevelMat2,qv2,destinationFile,False,l[0])
	
				print 'matrix_A.shape: ',matrix_A.shape
				print 'reduced dim: ',reducedDim
				continue
			if inmatrix :
				matrix_A.append([float(member) for member in l])


if __name__ == '__main__':
	#singularValueDec('mydata_level','mydata_doubleLevel_quan.data',1.0)
    singularValueDec('../data/dnn_484_512x5_3513','mydata_test.data',0.99)

