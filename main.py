from Src.perceptron import *
from Src.adaline import *
from Src.mlp import *
from Src.svm import *
from Src.hopfield import *
from Src.common import *
from Data.digits import *
from Data.random_models import getRandomArray

def get_algorithms():
    ans = -1
    while ans not in range(11):
        print('Perceptron.(0)\n' +
              'Adaline....(1)\n' +
              'LSQ........(2)\n' +
              'MLP........(3)\n' +
              'SOM1.......(4)\n' +
              'SOM2.......(5)\n' +
              'CP.........(6)\n' +
              'Hopfield...(7)\n' +
              'SVM........(8)\n' +
              'k-means....(9)\n' +
              'PCA........(10)')
        ans = int(input('Choose training algorithm: '))
    print()
    return ans

def get_training_or_recall():
    ans = -1
    while ans not in [0,1]:
        print('Training.(0)\n' +
              'Recall...(1)')
        ans = int(input('Choose training or recall: '))
    print()
    return ans

def get_data():
    ans = -1
    while ans not in range(8):
        print('Linearly separable....(0)\n' +
              'Center................(1)\n' +
              'Angle.................(2)\n' +
              'XOR...................(3)\n' +
              'Iris..................(4)\n' +
              'Housing...............(5)\n' +
              'XOR 3D................(6)\n' +
              'Linearly separable 3D (7)')
        ans = int(input('Choose training data: '))
    print()
    return ans

def get_beta():
    return float(input("Enter beta: "))

def get_epochs():
    return int(input("Enter epochs: "))

def get_digit():
    ans = -1
    while ans not in range(4):
        print('Digit 5.(0)\n' +
              'Digit 6.(1)\n' +
              'Digit 8.(2)\n' +
              'Digit 9.(3)')
        ans = int(input('Choose digit: '))
    print()
    return ans

def get_sync():
    ans = -1
    while ans not in range(2):
        print('Synchronous..(0)\n' +
              'Asynchronous.(1)')
        ans = int(input('Synchronous or Asynchronous recall: '))
    print()
    return ans

is_3D = lambda v: True if v in [6,7] else False 

data_files_train = ['Data/lin_sep.csv','Data/center.csv',
                    'Data/angle.csv','Data/xor.csv',
                    'Data/iris.data','Data/housing.data',
                    'Data/xor_3D.csv','Data/lin_sep_3D.csv']
data_files_recall = ['Data/lin_sep_recall.csv','Data/center_recall.csv',
                     'Data/angle_recall.csv','Data/xor_recall.csv',
                     'Data/iris.data','Data/housing.data',
                     'Data/xor_recall_3D.csv','Data/lin_sep_recall_3D.csv']

if __name__ == '__main__':
    while True:
        algorithm = get_algorithms()
        if algorithm == 0:
            is_training = True if get_training_or_recall() == 0 else False
            data = get_data()
            beta = get_beta()
            epochs = get_epochs()
            if is_training:
                if data == 4:     perceptron_flowers_train(data_files_train[data], beta, epochs)
                elif is_3D(data): perceptron_train_3D(data_files_train[data], beta, epochs)
                else:             perceptron_train_2D(data_files_train[data], beta, epochs)
            else:
                if data == 4:     perceptron_flowers_recall(data_files_recall[data], beta, epochs)
                elif is_3D(data): perceptron_recall_3D(data_files_train[data], data_files_recall[data], beta, epochs)
                else:             perceptron_recall_2D(data_files_train[data], data_files_recall[data], beta, epochs)
        elif algorithm == 1:
            is_training = True if get_training_or_recall() == 0 else False
            data = get_data()
            beta = get_beta()
            epochs = get_epochs()
            if is_training:
                if data == 4:     adaline_flowers_train(data_files_train[data], beta, epochs)
                elif is_3D(data): adaline_train_3D(data_files_train[data], beta, epochs)
                else:             adaline_train_2D(data_files_train[data], beta, epochs)
            else:
                if data == 4:     adaline_flowers_recall(data_files_recall[data], beta, epochs)
                elif is_3D(data): adaline_recall_3D(data_files_train[data], data_files_recall[data], beta, epochs)
                else:             adaline_recall_2D(data_files_train[data], data_files_recall[data], beta, epochs)
        elif algorithm == 2:
            print("Not implemented yet\n")
            #is_training = True if get_training_or_recall() == 0 else False
            #data = get_data()
            #beta = get_beta()
            #epochs = get_epochs()
            #if is_training:
            #    if data == 4:     lsq_flowers_train(data_files_train[data], beta, epochs)
            #    elif is_3D(data): lsq_train_3D(data_files_train[data], beta, epochs)
            #    else:             lsq_train_2D(data_files_train[data], beta, epochs)
            #else:
            #    if data == 4:     lsq_flowers_recall(data_files_recall[data], beta, epochs)
            #    elif is_3D(data): lsq_recall_3D(data_files_recall[data], beta, epochs)
            #    else:             lsq_recall_2D(data_files_recall[data], beta, epochs)
        elif algorithm == 3:
            is_training = True if get_training_or_recall() == 0 else False
            data = get_data()
            beta = get_beta()
            epochs = get_epochs()
            if is_training:
                if is_3D(data): mlp_train_3D(data_files_train[data], beta, epochs)
                else:           mlp_train_2D(data_files_train[data], beta, epochs)
            else:
                if is_3D(data): mlp_recall_3D(data_files_train[data], data_files_recall[data], beta, epochs)
                else:           mlp_recall_2D(data_files_train[data], data_files_recall[data], beta, epochs)
        elif algorithm == 4:
            print("Not implemented yet\n")
        elif algorithm == 5:
            print("Not implemented yet\n")
        elif algorithm == 6:
            print("Not implemented yet\n")
        elif algorithm == 7:
            is_training = True if get_training_or_recall() == 0 else False
            d = [i5,i6,i8,i9]
            if is_training: hopfield_train()
            else:
                is_sync = True if get_sync() == 0 else False
                hopfield_recall(d[get_digit()], is_sync)
        elif algorithm == 8:
            is_training = True if get_training_or_recall() == 0 else False
            data = get_data()
            if is_training:
                if is_3D(data): svm_train_3D(data_files_train[data])
                else:           svm_train_2D(data_files_train[data])
            else:
                if is_3D(data): svm_recall_3D(data_files_train[data])
                else:           svm_recall_2D(data_files_train[data])
        elif algorithm == 9:
            print("Not implemented yet\n")
        elif algorithm == 10:
            print("Not implemented yet\n")

        if input('Press anything to continue or x to stop:\n') == 'x': break
