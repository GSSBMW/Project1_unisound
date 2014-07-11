# Singular Value Decomposition

import numpy as np

def getDim(Sigma, threshold):
	"""Reduce the dimention of Sigma according to the treshold of Energy."""
	if not((threshold>=0.0) and (threshold<=1.0)):
		print "\tError: Threshold is illegal.\n"
		return 0
	
	sumEnergy = 0;
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

def levelMat2File(levelMatrix, fileName, firstColAsCon=True, activeStr='sigmoid'):
	"""Append parameter matrix of one level to a file.
	firsColAsCon is True means that get the first column 
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

def singularValueDec(sourceFile, destinationFile):
	"""Make singular value decompostion of all matrix in sourcefile,
	and write new marix of parameters to destinationFile in specified
	format.
	File format of each level:
	<affinetranform> 2 3
	 [
	  1 2 3
	  4 5 6 ]
	 [ 1 1 ]
	<sigmoid> 2 2
	"""
	f = open(sourceFile,'r')
	tmpf = open(destinationFile, 'w')
	tmpf.close()
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
				for i in range(len(matrix_A)-1):
					matrix_A[i].insert(0,matrix_A[len(matrix_A)-1][i])
				del matrix_A[len(matrix_A)-1]
				matrix_A = np.array(matrix_A)
				
				#process matrix
				U, Sigma, Vt = np.linalg.svd(matrix_A)
				threshold = 0.9
				reducedDim = getDim(Sigma, threshold)
				if reducedDim:
					U,Sigma,Vt = dimReduce(U, Sigma, Vt, reducedDim)
				
				#write matrix to file
				levelMat2File(np.dot(np.diag(Sigma),Vt),destinationFile)
				levelMat2File(U,destinationFile,False,l[0])
	
				print 'matrix_A.shapt: ',matrix_A.shape
				print 'reduced dim: ',reducedDim
				continue
			if inmatrix :
				matrix_A.append([float(member) for member in l])


if __name__ == '__main__':
	singularValueDec('mydata','mydata_doubleLevel.data')




