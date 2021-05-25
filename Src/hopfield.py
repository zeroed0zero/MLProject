from Data.digits import *
import random
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
import sys
import time

def hopfield(prototypes):
    synapses = np.zeros((prototypes[0].shape[0], prototypes[0].shape[0]))
    for p in prototypes:
        synapses += np.matrix(p).transpose() * np.matrix(p)
    return synapses

def hopfield_get_prototype_sync(synapses, prototype):
    output = np.ones(len(prototype)) * -1
    output_old = None
    for i in range(len(synapses)):
        v = -1 if sum([synapses[j,i]*prototype[j] for j in range(len(synapses))]) < 0 else 1
        output_old = np.copy(output)
        output[i] = v
        if not np.array_equal(output_old, output):
            print('Imperfect digit       Hopfield output')
            print(digits_to_str(prototype, output))
            time.sleep(.5)
    return output

def hopfield_get_prototype_async(synapses, prototype):
    output = np.ones(len(prototype)) * -1
    n = np.arange(len(synapses))
    random.shuffle(n)
    prototype_c = np.copy(prototype)
    output_old = None
    for i in n:
        v = -1 if sum([synapses[j,i]*prototype[j] for j in range(len(synapses))]) < 0 else 1
        output_old = np.copy(output)
        output[i] = v
        prototype[i] = v
        if not np.array_equal(output_old, output):
            print('Imperfect digit' + ' '*7 + 'Hopfield output')
            print(digits_to_str(prototype_c, output))
            time.sleep(.5)
    return output

def hopfield_train():
    np.set_printoptions(threshold=sys.maxsize)
    print(hopfield([p5, p6, p8, p9]))

def digit_convert(digit):
    digit = list(digit)
    digit_s = ''
    for i in range(len(digit)):
        if digit[i] == -1: digit_s += ' '
        else: digit_s += '@'
    return digit_s

def digits_to_str(digit1, digit2):
    digit1 = np.array(list(digit_convert(digit1)))
    digit2 = np.array(list(digit_convert(digit2)))

    digit1 = digit1.reshape((11,7))
    digit2 = digit2.reshape((11,7))
    digits = []
    for i in range(len(digit1)):
        digits.append(np.concatenate([digit1[i,], [' '*14], digit2[i,], ['\n']]))
    return "".join(np.ndarray.flatten(np.array(digits)))

def hopfield_recall_sync(digit):
    synapses = hopfield([p5, p6, p8, p9])
    hopfield_get_prototype_sync(synapses, digit)

def hopfield_recall_async(digit):
    synapses = hopfield([p5, p6, p8, p9])
    hopfield_get_prototype_async(synapses, digit)

def hopfield_recall(digit, is_sync):
    if is_sync: hopfield_recall_sync(digit)
    else: hopfield_recall_async(digit)
