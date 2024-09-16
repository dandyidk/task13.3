
import numpy as np
import struct
from keras import datasets
from sklearn.preprocessing import OneHotEncoder
from neuralnetwork import Neural_network

(train_images,test_images),(train_labels,test_labels)=datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], -1) / 255.0  # Flatten and normalize
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0      # Flatten and normalize

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
