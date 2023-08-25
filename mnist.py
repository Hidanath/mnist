import sys
import numpy as np
from keras.datasets import mnist

class Model:
    def __init__(self, imagesNum = 1000, alpha = 0.005, hidden_size = 100, batch_size = 100):
        (x_train, y_train), (x_test, y_test) = mnist.load_data() #Получение данных[x_train данные картинки(60000 матриц 28*28), y_train 60000 подписей к данным]
        self.images, labels = (x_train[0:imagesNum].reshape(imagesNum, 28*28) / 255, y_train[0:imagesNum]) #Матрица imagesNum на 784, imagesNum картинок со значениями силы для 784 пикселей от 0 до 1, imagesNum подписей
        self.labels = self.splitToClasses(labels)

        self.test_labels = self.splitToClasses(y_test)
        self.test_images = x_test.reshape(len(x_test),28*28) / 255 #imagesNum картинок со значениями силы для 784 пикселей от 0 до 1

        np.random.seed(1)

        self.tanh = lambda x: np.tanh(x)
        self.tanh2deriv = lambda x: 1 - (x**2)

        self.alpha, self.hidden_size, self.pixels_per_image, self.num_labels, self.batch_size = (alpha, hidden_size, 784, 10, batch_size) #num_labels количество классов

        self.weights_0_1 = 0.02*np.random.random((self.pixels_per_image, self.hidden_size)) - 0.01 #Матрица 784 на 100, тк вход 784 пикселя, скрытый слой 100 нейронов
        self.weights_1_2 = 0.2*np.random.random((self.hidden_size, self.num_labels)) - 0.1 #Матрица 100 на 10, тк скрытый слой 100 нейронов, выход 10 классов

    def softmax(self, x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)

    def splitToClasses(self, labels):
        """Conver labels to the class labels

        Args:
            labels (numpy.ndarray): Labels

        Returns:
            numpy.ndarray: Class labels
        """

        one_hot_labels = np.zeros((len(labels),10)) #Матрица imagesNum на 10
        for index, value in enumerate(labels): #Перезапись 1000 подписей в формате номер класса 1 из 10, к примеру one_hot_labels[0] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            one_hot_labels[index][value] = 1
            
        return one_hot_labels
    
    def learning(self, iterations:int = 300, status:bool = False, autosave:bool = True):
        """Learning model

        Args:
            iterations (int, optional): Number of training iterations. Defaults to 300.
            status (bool, optional): Displaying the training status in the console. Defaults to False.
            autosave (bool, optional): Autosave models weights. Defaults to True.
        """

        for j in range(iterations):
            correct_cnt = 0

            for i in range(int(len(self.images) / self.batch_size)): #Количество итераций зависит от размера пачки и количества картинок
                batch_start, batch_end=((i * self.batch_size),((i+1)*self.batch_size))

                layer_0 = self.images[batch_start:batch_end] #Срез по пачке
                layer_1 = self.tanh(np.dot(layer_0, self.weights_0_1))

                dropout_mask = np.random.randint(2, size=layer_1.shape) #Матрица нулей и единиц для dropout
                layer_1 *= dropout_mask * 2 #Отключение половины нейронов и уравнивание суммы

                layer_2 = self.softmax(np.dot(layer_1, self.weights_1_2))
                
                for k in range(self.batch_size): #Расчёт совпадения для каждого изображения из пачки
                    correct_cnt += int(np.argmax(layer_2[k:k+1]) == np.argmax(self.labels[batch_start+k:batch_start+k+1]))

                layer_2_delta = (self.labels[batch_start:batch_end]-layer_2) / (self.batch_size * layer_2.shape[0]) #Деление на размер пачки умноженный на количество строк layer_2 для усреднения
                layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * self.tanh2deriv(layer_1)

                layer_1_delta *= dropout_mask

                self.weights_1_2 += self.alpha * layer_1.T.dot(layer_2_delta)
                self.weights_0_1 += self.alpha * layer_0.T.dot(layer_1_delta)
            if status:
                sys.stdout.write("\r I:"+str(j)+\
                                " Train-Acc:" + str(correct_cnt/float(len(self.images))))
                
        if autosave:
            self.saveModel()

    def predict(self, image):
        """Neural netrwork prediction

        Args:
            image (numpy.ndarray): Pixel array

        Returns:
            numpy.ndarray: Class label
        """

        layer_0 = image
        layer_1 = self.tanh(np.dot(layer_0, self.weights_0_1))
        layer_2 = np.dot(layer_1, self.weights_1_2)

        return layer_2
    
    def checkAccuracy(self, imageCheck:int = 1000):
        """Check the performance of the model on test data

        Args:
            imageCheck (int, optional): Number of test images. Defaults to 1000.

        Returns:
            float: Model accuracy
        """

        images = self.test_images[0:imageCheck]
        correct = 0

        for i in range(len(images)):
            layer_0 = images[i:i+1]
            layer_1 = self.tanh(np.dot(layer_0, self.weights_0_1))
            layer_2 = self.softmax(np.dot(layer_1, self.weights_1_2))
            
            if np.argmax(self.test_labels[i:i+1]) == np.argmax(layer_2):
                correct += 1

        return correct / len(images)

    def saveModel(self):
        """Save the model to a file weights.npz"""

        np.savez("weights.npz", weights_0_1=self.weights_0_1, weights_1_2=self.weights_1_2)

    def loadModel(self):
        """Load a model from a file weights.npz

        Raises:
            FileNotFoundError: Model file not found
        """

        try:
            weights = np.load("weights.npz")
            self.weights_0_1 = weights["weights_0_1"]
            self.weights_1_2 = weights["weights_1_2"]

        except FileNotFoundError:
            raise FileNotFoundError("Model file not found")