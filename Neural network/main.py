
import numpy as np
import struct
from keras import datasets
from sklearn.preprocessing import OneHotEncoder
from neuralnetwork import Neural_network

train_images_name = r"train-images.idx3-ubyte"
test_images_name = r"t10k-images.idx3-ubyte"
train_labels_name = r"train-labels.idx1-ubyte"
test_labels_name = r"t10k-labels.idx1-ubyte"

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
        return images / 255.0  # Normalize pixel values

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

train_images = load_images(train_images_name)
test_images = load_images(test_images_name)
train_labels = load_labels(train_labels_name)
test_labels = load_labels(test_labels_name)

learn_rate = 0.01

network = Neural_network(784,20,10)



encoder = OneHotEncoder(sparse_output=False)
train_labels = encoder.fit_transform(train_labels.reshape(-1, 1))
test_labels = encoder.transform(test_labels.reshape(-1, 1))

for i in range(3):
    for img,label in zip(train_images,train_labels):
        img = np.array(img)
        img = img.reshape(784,1)
        label = np.array(label)
        label = label.reshape(10,1)
        network.training(img,label,learn_rate)
accuracy =0
correct =0
incorrect =0
for img , label in zip(test_images,test_labels):
    img = np.array(img)
    img = img.reshape(784, 1)
    label = np.array(label)
    label = label.reshape(10, 1)

    prediction_label = network.prediction(img)
    if prediction_label == np.argmax(label):
        correct +=1
    else: incorrect +=1
accuracy = 100*(correct/(correct+incorrect))
print(accuracy,"%")
