from perceptron import *

def show_algorithms():
    ans = -1
    while ans not in range(2):
        print('Perceptron (0)')
        print('Adaline (1)')
        ans = int(input('Choose training algorithm: '))
    return ans

def show_training_or_recall():
    ans = -1
    while ans not in [0,1]:
        print('Training (0)')
        print('Recall (1)')
        ans = int(input('Choose training or recall: '))
    return ans

def show_data():
    ans = -1
    while ans not in range(8):
        print('Linearly separable (0)')
        print('Center (1)')
        print('Angle (2)')
        print('XOR (3)')
        print('Iris (4)')
        print('Housing (5)')
        print('XOR 3D (6)')
        print('Linearly separable 3D (7)')
        ans = int(input('Choose training data: '))
    return ans

if __name__ == '__main__':
    while True:
        algorithm = show_algorithms()
        print('\n')
        data = show_data()
        print('\n')
        is_training = True if show_training_or_recall() == 0 else False
        is_2D = False if data in [6,7] else True 

        algorithm_train_2D = [perceptron_train_2D]
        algorithm_recall_2D = [perceptron_recall_2D]
        algorithm_train_3D = [perceptron_train_3D]
        algorithm_recall_3D = [perceptron_recall_3D]

        data_files = ['../Data/lin_sep.csv','../Data/center.csv','../Data/angle.csv','../Data/xor.csv','../Data/iris.data','../Data/housing.data','../Data/xor_3D.csv','../Data/lin_sep_3D.csv']

        if data == 4 and is_training:
            perceptron_flowers_training(data_files[data])
        elif data == 4 and not is_training:
            perceptron_flowers_recall(data_files[data])
        else:
            if is_2D: # 2D
                if is_training: algorithm_train_2D[algorithm](data_files[data])
                else: algorithm_recall_2D[algorithm](data_files[data])
            else: # 3D
                if is_training: algorithm_train_3D[algorithm](data_files[data])
                else: algorithm_recall_3D[algorithm](data_files[data])

        if input('Press anything to continue or x to stop: ') == 'x': break
        print('\n')
