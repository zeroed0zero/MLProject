import numpy as np
import random as rand
import csv

def random_float(low, high):
    return rand.random() * (high - low) + low

def getRandomArray(rand_start, rand_end, array_len):
    return [random_float(rand_start, rand_end) for i in range(int(array_len))]

def getLinearlySeparablePrototypes(proto_num):
    class0 = list(zip(getRandomArray(0, 0.3, proto_num/2),\
                      getRandomArray(0, 0.3, proto_num/2)))

    class1 = list(zip(getRandomArray(0.7, 0.9, proto_num/2),\
                      getRandomArray(0.7, 0.9, proto_num/2)))
    return tuple([class0, class1])

# Non linearly separable prototypes (angle)
def getAngle(proto_num):
    class0 = list(zip(getRandomArray(0, 0.3, proto_num/2),\
                      getRandomArray(0, 0.3, proto_num/2)))

    class1 = list(zip(getRandomArray(0,   0.3, proto_num/4) +\
                      getRandomArray(0.4, 0.9, proto_num/4),\
                      getRandomArray(0.4, 0.9, proto_num/4) +\
                      getRandomArray(0,   0.9, proto_num/4)))
    return tuple([class0, class1])

# Non linearly separable prototypes (Center)
def getCenter(proto_num):
    class0 = list(zip(getRandomArray(0.4, 0.6, proto_num/2),\
                      getRandomArray(0.4, 0.6, proto_num/2)))

    class1 = list(zip(getRandomArray(0,   0.9, proto_num/8) +\
                      getRandomArray(0,   0.9, proto_num/8) +\
                      getRandomArray(0,   0.3, proto_num/8) +\
                      getRandomArray(0.7, 0.9, proto_num/8),\
                      getRandomArray(0,   0.3, proto_num/8) +\
                      getRandomArray(0.7, 0.9, proto_num/8) +\
                      getRandomArray(0,   0.9, proto_num/8) +\
                      getRandomArray(0,   0.9, proto_num/8)))
    return tuple([class0, class1])

# Non linearly separable prototypes (XOR)
def getXOR(proto_num):
    class0 = list(zip(getRandomArray(0,   0.3, proto_num/4) +\
                      getRandomArray(0.7, 0.9, proto_num/4),\
                      getRandomArray(0,   0.3, proto_num/4) +\
                      getRandomArray(0.7, 0.9, proto_num/4)))

    class1 = list(zip(getRandomArray(0.7, 0.9, proto_num/4) +\
                      getRandomArray(0,   0.3, proto_num/4),\
                      getRandomArray(0,   0.3, proto_num/4) +\
                      getRandomArray(0.7, 0.9, proto_num/4)))
    return tuple([class0, class1])

def getLinearlySeparablePrototypes_3D(proto_num):
    class0 = list(zip(getRandomArray(0,   0.3, proto_num/2),\
                      getRandomArray(0,   0.3, proto_num/2),\
                      getRandomArray(0,   0.3, proto_num/2)))
    class1 = list(zip(getRandomArray(0.7, 0.9, proto_num/2),\
                      getRandomArray(0.7, 0.9, proto_num/2),\
                      getRandomArray(0.7, 0.9, proto_num/2)))
    return tuple([class0, class1])

def getXOR_3D(proto_num):
    class0 = list(zip(getRandomArray(0,   0.3, proto_num/4) +\
                      getRandomArray(0.7, 0.9, proto_num/4),\
                      getRandomArray(0,   0.3, proto_num/4) +\
                      getRandomArray(0.7, 0.9, proto_num/4),\
                      getRandomArray(0,   0.3, proto_num/4) +\
                      getRandomArray(0.7, 0.9, proto_num/4)))

    class1 = list(zip(getRandomArray(0.7, 0.9, proto_num/4) +\
                      getRandomArray(0,   0.3, proto_num/4),\
                      getRandomArray(0.7, 0.9, proto_num/4) +\
                      getRandomArray(0,   0.3, proto_num/4),\
                      getRandomArray(0,   0.3, proto_num/4) +\
                      getRandomArray(0.7, 0.9, proto_num/4)))
    return tuple([class0, class1])


def generateModel():
    print('Linearly Separable 0')
    print('Non Linearly Separable (Angle) 1')
    print('Non Linearly Separable (Center) 2')
    print('Non Linearly Separable (XOR) 3')
    print('Linearly Separable 3D 4')
    print('Non Linearly Separable 3D (XOR) 5')
    classes = int(input('Enter a number of the model you want: '))
    proto_num = int(input('Enter the number of prototypes you want (power of 2): '))
    file_name = input('Enter a file name: ')

    c1, c2 = None, None
    if classes == 0:
        c1, c2 = getLinearlySeparablePrototypes(proto_num)
    elif classes == 1:
        c1, c2 = getAngle(proto_num)
    elif classes == 2:
        c1, c2 = getCenter(proto_num)
    elif classes == 3:
        c1, c2 = getXOR(proto_num)
    elif classes == 4:
        c1, c2 = getLinearlySeparablePrototypes_3D(proto_num)
    elif classes == 5:
        c1, c2 = getXOR_3D(proto_num)

    writeClassesToFile([c1, c2], file_name)

def writeClassesToFile(classes, file_name):
    with open(file_name, 'w') as of:
        writer = csv.writer(of)
    
        for i in range(len(classes)):
            for j in range(len(classes[i])):
                proto = []
                for k in range(len(classes[i][j])):
                    proto.append(classes[i][j][k])
                proto.append(i)
                writer.writerow(proto)

if __name__ == '__main__':
    generateModel()
