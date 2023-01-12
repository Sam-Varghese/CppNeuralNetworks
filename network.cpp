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

    double linear(){
        return this->z;
    }

    //* Getting derivative values of activation functions
    
    double sigmoidDerivative(){
        // Assuming that the output already contains the sigmoid of z
        // Formula = g(z) * (1 - g(z))
        this->derivativeOutput = this->output * (1 - this->output);
        return this->derivativeOutput;
    }

    int reLUDerivative(){
        // 0 if x < 0 and 1 if x >= 0
        if(this->output < 0){
            this->derivativeOutput = 0;
        } else {
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
    vector<double> layerOutput, biases;
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
            predictions.back() += this->biases[output];
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

    void collectBiases()
    {
        for (int i = 0; i < this->neuronCount; i++)
        {
            this->biases.push_back(this->neuronCollection[i].getBias());
        }
    }
};

class NeuralNetwork
{
private:
    vector<int> networkArchitecture;

public:
    NeuralNetwork(vector<int> inputNetworkArchitecture){
        this->networkArchitecture = inputNetworkArchitecture;
    }
};

int main(void)
{

    return 0;
}