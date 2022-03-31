#pragma once
#include "MNIST.hpp"

// normalizeImages "normalizes" images so that smallest pixel values are zero, and
// largest pixel values are 1.0f.  for example, if an image has pixel values of
// [1, 2, 3, 4, 5] (5 being the largest), the resultant pixel values are [0.0, 0.25, 0.5, 0.75, 1.0].
// i.e. all pixel values are subtracted from the smallest pixel value and then divided by the larest pixel value in the same image.
void normalizeImages(MNISTImages &images);

// oneHotEncoding converts/prepares output feature vector using one-hot encoding scheme.  The result
// shoudl eb stored somewhere inside the MNISTLabels struct.
void oneHotEncoding(MNISTLabels &labels);

// This declares the prototype for the Sigmoid function.
float sigmoid(float x);

// This is to create an alias to the function-pointer data type.
using ActivationFunction = float (*)(float);

// I created a Layer struct to store layer-related information
// (input size, output size, weights, biases, output, and activation function.)
// It should be noted that you should allow each layer to use a different activation function.
struct Layer
{
    int isize;
    int osize;
    float *weight;
    float *biases;
    float *output;
    float *theta;
    ActivationFunction acts;
};

// I created this struct Model to store model information (e.g. number of layers, details about each layer)
struct Model
{
    Layer *layer_d;
    int n_layers;
};

// buildModels allocates memory and create all layers of NN based on given parameters:
//      noLayers: number of layers (of neurons)
//      layerSizes: a 1-D array contains the number of neurons at each layer.
//      aFunctions: a 1-D array contains activation function pointers used in each layer.
Model buildModel(const int noLayers, const int *layerSizes, ActivationFunction *aFunctions);

// destroyModel de-allocates allocated memory in buildModel.
void destroyModel(Model &);

// initializeModel initializes all weights in all layers.
// The weights are uniformly distributed random numbers in [-1, 1].
void initializeModel(Model &model);

// testWeights function initialize weights according to the Excel spreadsheets given to you
// so that it will be easy for you to debug/test your implementation.
void testWeights(Model &);

// feedForward carries out the feed-forward process to do inferencing.
void feedForward(const float *input, Model &model, float *output);

// backPropagate conducts back-propagation to adjust/learn proper weights so that
// outputs match given inputs. The function returns SSE (sum of squared error).
float backPropagate(float *target, Model &model, float *input, float learningRate = 0.001f);
