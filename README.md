# task13.3

# Preceptrons
They are artifical neural network that can take multiple binary input, with each inputs having weights indicating the significance of this input, and outputs one binary input.

The equation of the binary output is as follows

```math
output = 0 if w⋅x+b≤0 
```
```math
output = 1 if w⋅x+b>0
```
But due to how sensitive they are where minimal change of weight of input can drastically change the output, its preferable to use sigmoid neuron

# sigmoid neuron
Just like preceptron it has multiple inputs and one outputs but the difference is that this inputs can range from 0 to 1 and the output is the sigmoid function which is equal to:

```math
σ(w⋅x+b)
```
where
```math
σ(z)≡\frac{1}{1+e−z}
```
# Gradient descent 
Its an optimization algorithm used to minimize a cost function C. When making changes in the parameters of C 

# Program
To make a handwritten numbers recognition neural network, we need to take the dataset and turn each image to a vector column.

Since we are using MNSIT dataset, the images will be 28x28 which means there will be 784 pixel to be read, meaning that our neural network will take 784 input, and the number of output layer of neurons will be 10 as we would be using a 10 output encoding

## Forward propagation

After turning the images and labels to column vectors we are going to use the sigmoid function to calculate the output from the input to the hidden layer with the currently assigned random weights and the bias set to 0, then recalculate from hidden to output.

Of course the output will have errors and so we are going to use the following cost function:

```math
C(w,b)≡\frac{1}{2n}\sum_{x}{}∥y(x)−a∥^{2}.
```

## Backward propagation and gradient descent

Gradient descent is an

