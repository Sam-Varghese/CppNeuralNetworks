# Neural Network Concepts

This section will cover the basic concepts of neural networks which I've learnt in recent few months. We will start with the basic building blocks of neural networks, and then move on to more advanced concepts.

> There might be some mistakes in the concept I've on my mind. I encourage the viewers to point out my mistakes so I could brush up my concepts.

## Neurons

[Neuron](https://www.wikiwand.com/en/Neuron) is the basic building block of all neural networks.

For programming the neurons, each neurons will have the following parameters associated with them

1. **Weight**: When *N* neurons connect with *M* other neurons (fully connected neural network), then each of the *N* neurons will have *M* different weights for connecting with next *M* neurons.
2. **Bias**: All neurons (Except those which belong to the output layer) will have a *double* value *Bias* associated with it.
3. **Activation**: All the neurons have an activation value associated with them. Along with the weights and biases, a final decision is made wether to fire up the output neuron or not.
4. **Activation Function**: Activation function is a function which processes the activation value, and outputs a new result. There are many activation functions like [Sigmoid](https://www.wikiwand.com/en/Sigmoid_function), [ReLU](https://www.wikiwand.com/en/Rectifier_(neural_networks)), [etc](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

![Neuron Image](https://miro.medium.com/max/1400/1*hkYlTODpjJgo32DoCOWN5w.png)

## Layers

- Layers are the collection of [Neurons](https://www.wikiwand.com/en/Neuron).
- They can be divided into 3 categories:

1. Input Layer: The very 1st layer of Neural Network.
2. Hidden Layers: All the layers between Input and Output layers.
3. Output Layer: The final layer of Neural Network.

![Layer image](https://stackabuse.s3.amazonaws.com/media/deep-learning-in-keras-building-a-deep-learning-model-1.png)

## Activation Functions

### Sigmoid

f(x) = 1/(1 + e<sup>-x</sup>)

### ReLU

`f(x) = max(0, x)`

## Feed Forward Neural Network

- Suppose there are *N* neurons in first layer, and *M* neurons in the 2nd layer, and the neural network is a fully connected one.
- Weights of the connection 1st neuron will form with rest *M* neurons be w<sub>1</sub><sup>1</sup>, w<sub>2</sub><sup>1</sup>, w<sub>3</sub><sup>1</sup>, w<sub>4</sub><sup>1</sup>