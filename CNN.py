import numpy as np
import struct
import gzip
import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)
        self.bias = torch.zeros(out_channels, requires_grad=True)

    def forward(self, x):
        self.input = x
        return F.conv2d(x, self.weights, self.bias, stride=self.stride, padding=self.padding)

    def backward(self, grad_output, learning_rate):
        """
        Backward pass to calculate gradients and update weights.
        grad_output: The gradient of the loss with respect to the output of this conv layer.
        """

        grad_input = F.conv_transpose2d(grad_output, self.weights, stride=self.stride, padding=self.padding)
        grad_weights = F.conv2d(self.input.permute(1, 0, 2, 3), grad_output.permute(1, 0, 2, 3), padding=self.padding).permute(1, 0, 2, 3)
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        with torch.no_grad():
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias

        return grad_input

def maxpool_backward(input, grad_output, kernel_size=2, stride=2):
    """
    Backward pass for MaxPooling. It propagates the gradient to the max values
    from the previous forward pass.
    
    Args:
    - input: Input tensor (before max-pooling), shape: (batch_size, channels, height, width)
    - grad_output: Gradient passed from the next layer, shape: (batch_size, channels, out_height, out_width)
    - kernel_size: Size of the max-pooling window (default is 2x2)
    - stride: Stride of the max-pooling operation (default is 2)
    
    Returns:
    - grad_input: The gradient for the input of the max-pooling layer.
    """

    _, indices = F.max_pool2d(input, kernel_size=kernel_size, stride=stride, return_indices=True)
    grad_input = torch.zeros_like(input)
    grad_input = F.max_unpool2d(grad_output, indices, kernel_size=kernel_size, stride=stride, output_size=input.size())
    
    return grad_input

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
        self.conv1 = Conv2D(1, 3, 5)  # 3 filters, 5x5 kernel, 1 input channel
        self.maxpool1 = MaxPool2d(2,2)  # 2x2 pooling
        self.conv2 = Conv2D(3, 1, 5)  # 1 filter, 5x5 kernel, 3 input channels
        self.maxpool2 = MaxPool2d(2,2)  # 2x2 pooling
        self.linear1 = Linear(16, hidden_size)
        self.linear2 = Linear(hidden_size, output_size)

    def forward(self, X):
        # Input X is of shape (batch_size, 1, 28, 28)
        output = self.conv1.forward(np.array(X.reshape(-1, 1, 28, 28)))  # Conv2D expects 4D input
        self.maxpool1_input = output
        output = self.maxpool1.forward(output)
        output = self.conv2.forward(output)
        self.maxpool2_input = output
        output = self.maxpool2.forward(output)
        
        # Flatten the output from maxpool2 before feeding into the linear layers
        output = output.reshape(output.shape[0], -1)  # Flatten to (batch_size, num_features)
        
        output = self.linear1.forward(output)
        output = self.activation_function.forward(output)
        output = self.linear2.forward(output)
        return output

    def backward(self, X, Y, Y_pred, learning_rate):
        # Compute gradient of loss with respect to the prediction
        output_gradient = Y_pred - Y
        
        # Backprop through the linear layers
        output_gradient = self.linear2.backward(output_gradient, learning_rate)
        output_gradient = self.activation_function.backward(output_gradient)
        output_gradient = self.linear1.backward(output_gradient, learning_rate)
        
        # Reshape gradient to match the output of maxpool2 (reverse the flattening)
        output_gradient = output_gradient.reshape(-1, 1, 4, 4)  # (batch_size, 1, 4, 4)
        
        # Backprop through the convolutional and pooling layers
        output_gradient = maxpool_backward(self.maxpool2_input, output_gradient)  # No learning rate needed for MaxPool2D
        output_gradient = self.conv2.backward(output_gradient, learning_rate)
        output_gradient = maxpool_backward(self.maxpool1_input, output_gradient)  # No learning rate for MaxPool2D
        output_gradient = self.conv1.backward(output_gradient, learning_rate)

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

X_train = load_mnist_images(r"C:\Users\sahil\Desktop\Secured+\data\Training Set\MNIST\raw\train-images-idx3-ubyte.gz")
Y_train = load_mnist_labels(r"C:\Users\sahil\Desktop\Secured+\data\Training Set\MNIST\raw\train-labels-idx1-ubyte.gz")

X_test = load_mnist_images(r"C:\Users\sahil\Desktop\Secured+\data\Testing Set\MNIST\raw\t10k-images-idx3-ubyte.gz")
Y_test = load_mnist_labels(r"C:\Users\sahil\Desktop\Secured+\data\Testing Set\MNIST\raw\t10k-labels-idx1-ubyte.gz")

nn = NeuralNetwork(28*28, 100, 10)
nn.train(X_train, Y_train, X_test, Y_test, 25, 0.01, 32)