#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <time.h>

using namespace std;

class ActivationFunctions
{
private:
    double output;
    // Value after putting z into derivative of the activation function
    double derivativeOutput;
    double z;

public:
    void setZ(double zInput)
    {
        this->z = zInput;
    }

    double reLU()
    {
        this->output = max(0.0, this->z);
        return this->output;
    }

    double sigmoid()
    {
        // Performing 1/1+exp(-z)
        this->output = 1.0 / (1 + exp(this->z * -1));
        return this->output;
    }

    double linear()
    {
        return this->z;
    }

    //* Getting derivative values of activation functions

    double sigmoidDerivative()
    {
        // Assuming that the output already contains the sigmoid of z
        // Formula = g(z) * (1 - g(z))
        this->derivativeOutput = this->output * (1 - this->output);
        return this->derivativeOutput;
    }

    int reLUDerivative()
    {
        // 0 if x < 0 and 1 if x >= 0
        if (this->output < 0)
        {
            this->derivativeOutput = 0;
        }
        else
        {
            this->derivativeOutput = 1;
        }

        return this->derivativeOutput;
    }
};

class Neuron
{
private:
    vector<double> weight;
    // connection count store the number of neurons in the next layer so it can make that many random weights and biases
    int input, connectionCount;
    double prediction = 0.0, bias;

public:
    Neuron(int inputConnectionCount)
    {
        this->connectionCount = inputConnectionCount;
        srand((unsigned)time(NULL));
        // Each neuron will initialize the weights randomly for the output neurons
        for (int i = 0; i < this->connectionCount; i++)
        {
            this->weight[i] = (double)rand() / RAND_MAX;
        }
        this->bias = (double)rand() / RAND_MAX;
    }

    // Activation here refers to the value neuron holds ie input
    void setActivation(int inputValue)
    {
        this->input = inputValue;
    }

    vector<double> getWeights()
    {
        return this->weight;
    }

    int getBias()
    {
        return this->bias;
    }
};

class Layer
{
private:
    // LayerIndex keeps a unique index for all layers so they can be differentiated
    // outputLayer count keeps a count of number of neurons that comes after this layer so it can initiate that many weights and biases
    int neuronCount, layerIndex, outputLayerCount;
    // A layer should know the values of all biases of next layer so as to computer w*x + b, as b here refers to bias of next layer
    vector<double> layerOutput, currentLayerBiases, nextLayerBiases;
    vector<Neuron> neuronCollection;
    vector<double> activationValues;
    vector<vector<double>> weights;
    string activation;

public:
    Layer(int inputLayerNeuronCount, int layerIndex, int inputOutputLayerNeuronCount, string inputActivation)
    {
        this->neuronCount = inputLayerNeuronCount;
        this->outputLayerCount = inputOutputLayerNeuronCount;
        this->activation = inputActivation;
        // Appending neurons to this layer
        for (int i = 0; i < this->neuronCount; i++)
        {
            this->neuronCollection.push_back(Neuron(this->outputLayerCount));
        }
        // Storing the randomly generated weights of each neuron in weights 2D array
        collectWeights();
    }

    void setNeuronActivations(vector<double> inputActivationValues)
    {
        this->activationValues = inputActivationValues;
        for (int i = 0; i < this->neuronCount; i++)
        {
            this->neuronCollection[i].setActivation(inputActivationValues[i]);
        }
    }

    vector<double> computeLayerOutput()
    {
        // Performs w*x + b and returns a vector of size equal to neurons count in the output layer
        vector<double> predictions;
        ActivationFunctions activationFunctions;
        for (int output = 0; output < this->outputLayerCount; output++)
        {
            predictions.push_back(0.0);
            for (int i = 0; i < this->neuronCount; i++)
            {
                predictions.back() += this->weights[i][output] * this->activationValues[i];
            }
            predictions.back() += this->nextLayerBiases[output];
        }
        // Applying activation functions
        if (this->activation == "relu")
        {
            for (int i = 0; i < predictions.size(); i++)
            {
                activationFunctions.setZ(predictions[i]);
                predictions[i] = activationFunctions.reLU();
            }
        }
        else if (this->activation == "sigmoid")
        {
            for (int i = 0; i < predictions.size(); i++)
            {
                activationFunctions.setZ(predictions[i]);
                predictions[i] = activationFunctions.sigmoid();
            }
        }

        this->layerOutput = predictions;
        return this->layerOutput;
    }

    void collectWeights()
    {
        for (int i = 0; i < this->neuronCount; i++)
        {
            this->weights.push_back(this->neuronCollection[i].getWeights());
        }
    }

    void setNextLayerBiases(vector<double> nextLayerBiasesInput)
    {
        // nth layers uses the bias values of (n+1)th layer
        this->nextLayerBiases = nextLayerBiasesInput;
    }
    void setCurrentLayerBiases()
    {
        for (int i = 0; i < this->neuronCount; i++)
        {
            this->currentLayerBiases.push_back(this->neuronCollection[i].getBias());
        }
    }
    vector<double> getCurrentLayerBiases()
    {
        return this->currentLayerBiases;
    }
};

class NeuralNetwork
{
private:
    vector<int> networkArchitecture;
    vector<string> activations;
    vector<Layer> layers;
    vector<double> activationValues;

    /* For all the training example, store the activation values that were calculated (of all layers), so that backpropagation may be implemented. It's a 3D vector, so it's elements will

        [
             [ [layer1], [layer2] ], 1st training example values
             [ [layer1], [layer2] ], 2nd training example values
             ...
        ]
    */
    vector<vector<vector<double>>> activationCompilation;

public:
    NeuralNetwork(vector<int> inputNetworkArchitecture, vector<string> layerActivations)
    {

        // Appending a 0 in the end of architecture to denote that it doesn't have any layer afterwards, hence this is the last layer.
        inputNetworkArchitecture.push_back(0);

        this->networkArchitecture = inputNetworkArchitecture;
        this->activations = layerActivations;

        // Creating layers in the neural network

        for (int i = 0; i < inputNetworkArchitecture.size(); i++)
        {
            layers.push_back(Layer(this->networkArchitecture[i], i, this->networkArchitecture[i + 1], this->activations[i]));
        }

        // Informing the layers, biases of neurons of next layer for enabling them to computer w*x + b

        for (int i = 0; i < inputNetworkArchitecture.size() - 1; i++)
        {
            this->layers[i].setNextLayerBiases(this->layers[i + 1].getCurrentLayerBiases());
        }
    }
    bool setActivationValues(vector<double> inputActivationValues)
    {
        if (inputActivationValues.size() != this->networkArchitecture[0])
        {
            cout << "Error: Activation size doesn't match network topology.";
            return false;
        }

        this->layers[0].setNeuronActivations(inputActivationValues);

        return true;
    }

    vector<double> startFeedForward()
    {
        // Returns the activations of output layer
        vector<vector<double>> finalActivations;
        vector<double> layerOutput;
        // Executing a loop for feed forward mechanism, but this loop will not run for the last layer because it needs to be handled a bit differently
        for (int i = 0; i < this->layers.size() - 1; i++)
        {
            layerOutput = this->layers[i].computeLayerOutput();
            finalActivations.push_back(layerOutput);
            this->layers[i].setNeuronActivations(layerOutput);
        }
        // Pushing all the activation values of all layers into the activationCompilation 3D vector
        this->activationCompilation.push_back(finalActivations);
        return layerOutput;
    }
};

int main(void)
{

    return 0;
}