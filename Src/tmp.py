import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split

data = read_csv('../Data/iris.data')


x=data[[0,1,2,3]].values
lables = data[4].values
map_dict = {'Iris-setosa':0,
            'Iris-versicolor':1,
            'Iris-virginica':2}
t = np.array([map_dict[lbl] for lbl in labels])

# Split randomly the dataset into train and test sets
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3)
iris_setosa = [data[[2,3]][i] for i in len(data[[2,3]].values) if data[4][i] == 'Iris-setosa']
print(iris_setosa)
iris_versicolor = [[],[]]
iris_virginica = [[],[]]
for i in range(len(csv)):
    if csv[i][4] == 'Iris-setosa':
        iris_setosa[0].append(csv[i][2:4][0])
        iris_setosa[1].append(csv[i][2:4][1])
    elif csv[i][4] == 'Iris-versicolor':
        iris_versicolor[0].append(csv[i][2:4][0])
        iris_versicolor[1].append(csv[i][2:4][1])
    else: # Iris-virginica
        iris_virginica[0].append(csv[i][2:4][0])
        iris_virginica[1].append(csv[i][2:4][1])

plt.xlabel('Petal Length (in cm)')
plt.ylabel('Petal Width (in cm)')
plt.plot(iris_setosa[0], iris_setosa[1], 'ro')
plt.plot(iris_versicolor[0], iris_versicolor[1], 'bo')
plt.plot(iris_virginica[0], iris_virginica[1], 'go')
plt.show()
