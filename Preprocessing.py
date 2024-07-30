import torch
from torchvision import datasets, transforms

import pickle

def load_and_preprocess_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) #Maybe modify these numbers?
    ])

    #Modify root directory for datasets
    #Figure out how images will be formatted, modify method as needed
    train_dataset = datasets.MNIST(root=r"C:\Users\timot\Desktop\Secured+\data\Training Set", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=r"C:\Users\timot\Desktop\Secured+\data\Testing Set", train=False, download=True, transform=transform)


    train_images = train_dataset.data.float() / 255.0
    train_images = train_images.reshape(-1, 28*28).numpy()
    train_labels = torch.nn.functional.one_hot(train_dataset.targets, num_classes=10).numpy()

    test_images = test_dataset.data.float() /255.0
    test_images = test_images.reshape(-1, 28*28).numpy()
    test_labels = torch.nn.functional.one_hot(test_dataset.targets, num_classes=10).numpy()

    return train_images, train_labels, test_images, test_labels

def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

train_images, train_labels, test_images, test_labels = load_and_preprocess_mnist()


save_to_pickle(train_images, 'train_images.pkl')
save_to_pickle(train_labels, 'train_labels.pkl')
save_to_pickle(test_images, 'test_images.pkl')
save_to_pickle(test_labels, 'test+labels.pkl')