import numpy as np
import struct
import gzip

class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, np.transpose(self.weights))
        weight_gradient = np.dot(np.transpose(self.input), output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        self.weights = self.weights - learning_rate * weight_gradient
        self.bias = self.bias - learning_rate * bias_gradient

        return input_gradient

class Sigmoid:
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)


#Could potentially add more hidden layers for larger model/more accurate results
#Ex: For 3 hidden layers, __init__ method
#self.linear1 = Linear(input_size, hidden_size)
#self.linear2 = Linear(hidden_size, hidden_size)
#self.linear3 = Linear(Hidden_size, hidden_size)
#self.linear4 = Linear(hidden_size, output_size)

#Forward method:
#output = self.linear1.forward(X)
#output = self.activation_function.forward(output)
#output = self.linear2.forward(output)
#output = self.activation_function.forward(output)
#output = self.linear3.forward(output)
#output = self.activation_function.forward(output)
#output = self.linear4.forward(output)
#return output

#Same idea for the backward method

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.activation_function = Sigmoid()

        self.linear1 = Linear(input_size, hidden_size)
        self.linear2 = Linear(hidden_size, output_size)

    def forward(self, X):
        output = self.linear1.forward(X)
        output = self.activation_function.forward(output)
        output = self.linear2.forward(output)
        return output
    
    def backward(self, X, Y, Y_pred, learning_rate):
        output_gradient = Y_pred - Y
        output_gradient = self.linear2.backward(output_gradient, learning_rate)
        output_gradient = self.activation_function.backward(output_gradient)
        output_gradient = self.linear1.backward(output_gradient, learning_rate)

    def test(self, X, Y):
        true_positives = np.zeros(10)
        false_positives = np.zeros(10)
        false_negatives = np.zeros(10)
        correct, total = 0, 0

        for i in range(len(X)):
            Y_pred = np.argmax(self.forward(X[i:i+1]))
            Y_actual = np.argmax(Y[i])

            if Y_pred == Y_actual:
                correct += 1
                true_positives[Y_actual] += 1
            else:
                false_positives[Y_pred] += 1
                false_negatives[Y_actual] += 1
            total += 1

        accuracy = correct / total
        precision = np.mean(true_positives / (true_positives + false_positives + 1e-10))
        recall = np.mean(true_positives / (true_positives + false_negatives + 1e-10))

        return accuracy, precision, recall

    def train(self, X_train, Y_train, X_test, Y_test, epochs, learning_rate, batch_size):
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, Y_batch, output, learning_rate)
            
            train_acc, train_prec, train_rec = self.test(X_train, Y_train)
            test_acc, test_prec, test_rec = self.test(X_test, Y_test)

            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
            print(f"Epoch {epoch+1}: Test Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
            print()
            learning_rate *= 0.99  # Learning rate decay

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Read metadata
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return images / 255.0

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # Read metadata
        magic, num = struct.unpack(">II", f.read(8))
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        # One-hot encode labels
        return np.eye(10)[labels]

X_train = load_mnist_images(r"C:\Users\timot\Desktop\Secured+\data\Training Set\MNIST\raw\train-images-idx3-ubyte.gz")
Y_train = load_mnist_labels(r"C:\Users\timot\Desktop\Secured+\data\Training Set\MNIST\raw\train-labels-idx1-ubyte.gz")

X_test = load_mnist_images(r"C:\Users\timot\Desktop\Secured+\data\Testing Set\MNIST\raw\t10k-images-idx3-ubyte.gz")
Y_test = load_mnist_labels(r"C:\Users\timot\Desktop\Secured+\data\Testing Set\MNIST\raw\t10k-labels-idx1-ubyte.gz")

nn = NeuralNetwork(28*28, 100, 10)
nn.train(X_train, Y_train, X_test, Y_test, 25, 0.01, 32)
