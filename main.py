from Src.perceptron import *
from Src.adaline import *
from Data.random_models import getRandomArray

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

def get_beta():
    return float(input("Enter beta: "))

def get_epochs():
    return int(input("Enter epochs: "))

if __name__ == '__main__':
    while True:
        algorithm = show_algorithms()
        print()
        is_training = True if show_training_or_recall() == 0 else False
        print()
        data = show_data()
        print()
        is_2D = False if data in [6,7] else True 

        algorithm_train_2D = [perceptron_train_2D, adaline_train_2D]
        algorithm_recall_2D = [perceptron_recall_2D]
        algorithm_train_3D = [perceptron_train_3D]
        algorithm_recall_3D = [perceptron_recall_3D]

        data_files_train = ['Data/lin_sep.csv','Data/center.csv','Data/angle.csv','Data/xor.csv','Data/iris.data','Data/housing.data','Data/xor_3D.csv','Data/lin_sep_3D.csv']
        data_files_recall = ['Data/lin_sep_recall.csv','Data/center_recall.csv','Data/angle_recall.csv','Data/xor_recall.csv','Data/iris.data','Data/housing.data','Data/xor_recall_3D.csv','Data/lin_sep_recall_3D.csv']
        beta = get_beta()
        epochs = get_epochs()

        if data == 4 and is_training:
            perceptron_flowers_training(data_files_train[data], np.array(getRandomArray(-0.5, 0.5, 3)), beta, epochs)
        elif data == 4 and not is_training:
            perceptron_flowers_recall(data_files_train[data], np.array(getRandomArray(-0.5, 0.5, 3)), beta, epochs)
        else:
            if is_2D:
                if is_training: algorithm_train_2D[algorithm](data_files_train[data], np.array(getRandomArray(-0.5, 0.5, 3)), beta, epochs)
                else: algorithm_recall_2D[algorithm](data_files_train[data], data_files_recall[data], np.array(getRandomArray(-0.5, 0.5, 3)), beta, epochs)
            else:
                if is_training: algorithm_train_3D[algorithm](data_files_train[data], np.array(getRandomArray(-0.5, 0.5, 4)), beta, epochs)
                else: algorithm_recall_3D[algorithm](data_files_train[data], data_files_recall[data], np.array(getRandomArray(-0.5, 0.5, 4)), beta, epochs)

        if input('Press anything to continue or x to stop: ') == 'x': break
        print('#'*50 + '\n')
