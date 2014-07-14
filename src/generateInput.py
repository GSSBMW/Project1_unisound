
import random

def generateInput(filename, dimention, amount):
    """Generate input vectors of one level in dnn, and write to file.
    filename    Name file to store data
    dimention   Specify dimention of each input vector
    amount      Specify number of vectors which need to be generated
    """
    f = open(filename, 'w')
    for i in range(amount):
        f.write('[ ')
        for j in range(dimention):
            f.write('%.6f '%random.random())
        f.write(']\n')


