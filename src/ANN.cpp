#include "ANN.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
using namespace std;

// TODO: use one-hot encoding to encode the targets
void oneHotEncoding(MNISTLabels &labels)
{
    labels.codes = new float *[labels.noItems];
    for (int i = 0; i < labels.noItems; i++)
    {
        labels.codes[i] = new float[10];
        for (int j = 0; j < 10; j++)
        {
            labels.codes[i][j] = 0;
            if (labels.labels[i] == j)
            {
                labels.codes[i][j] = 1;
            }
        }
    }
}

// TODO: normalize images so each pixel is stored using float (fp32, float32)
// and each image's pixel values are between 0.0f - 1.0f
void normalizeImages(MNISTImages &images)
{
    images.normalizedImages = new float *[images.noItems];
    for (int i = 0; i < images.noItems; i++)
    {
        images.normalizedImages[i] = new float[images.rows * images.cols];
#pragma omp parallel for
        for (int j = 0; j < (images.rows * images.cols); j++)
        {
            images.normalizedImages[i][j] = float(images.images[i][j]) / 255.0;
        }
    }
}
// okay, this one is free, I am pretty sure you can do this...
float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

// TODO: build your NN model based on the passed in parameters. Hopefully all memory allocation is done here.
Model buildModel(const int noLayers, const int *layerSizes, ActivationFunction *acts)
{
    Model model;
    model.n_layers = noLayers;
    model.layer_d = new Layer[noLayers - 1];
    model.layer_d[0].isize = layerSizes[0]; // 2
    model.layer_d[0].osize = layerSizes[1]; // 3
    model.layer_d[0].acts = acts[0];
    model.layer_d[1].isize = layerSizes[1]; // 3
    model.layer_d[1].osize = layerSizes[2]; // 2
    model.layer_d[1].acts = acts[1];
    return model;
}

// TODO: de-allocate all allocated memory (hint: each new must comes with a delete)
void destroyModel(Model &model)
{
    // delete[] model.layer_d;
    // cout << "\nDon't forget to implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}
// TODO: initialize your model with uniformly distributed random numbers between [-1.0f, 1.0f] for your  weights and biases.
void initializeModel(Model &model)
{
    model.layer_d[0].weight = new float[model.layer_d[0].isize * model.layer_d[0].osize];
    model.layer_d[0].biases = new float[model.layer_d[0].osize];
    model.layer_d[1].weight = new float[model.layer_d[1].isize * model.layer_d[1].osize];
    model.layer_d[1].biases = new float[model.layer_d[1].osize];
    random_device rand;
    mt19937 generator(rand());
    uniform_real_distribution<float> distribute(-1.0f, 1.0f);
#pragma omp parallel for
    for (int no_las = 0; no_las < model.n_layers - 1; no_las++) // 2
    {
        for (int o = 0; o < model.layer_d[no_las].osize; o++) // 3 2
        {
            model.layer_d[no_las].biases[o] = distribute(generator);
        }
    }
#pragma omp parallel for
    for (int no_las = 0; no_las < model.n_layers - 1; no_las++) // 2
    {
        for (int i = 0; i < model.layer_d[no_las].isize * model.layer_d[no_las].osize; i++) // 2 3
        {
            model.layer_d[no_las].weight[i] = distribute(generator);
        }
    }
}

// TODO: testWeights function initialize weights according to the Excel spreadsheets given to you so that it will be easy for you to debug/test your implementation.
void testWeights(Model &model)
{

    model.layer_d[0].weight[0] = 0.1;
    model.layer_d[0].weight[1] = 1.1;
    model.layer_d[0].weight[2] = 0.2;
    model.layer_d[0].weight[3] = 1.2;
    model.layer_d[0].weight[4] = 0.3;
    model.layer_d[0].weight[5] = 1.3;

    model.layer_d[0].biases[0] = -1;
    model.layer_d[0].biases[1] = -2;
    model.layer_d[0].biases[2] = -3;

    model.layer_d[1].weight[0] = -0.1;
    model.layer_d[1].weight[1] = -0.3;
    model.layer_d[1].weight[2] = -0.5;
    model.layer_d[1].weight[3] = -0.2;
    model.layer_d[1].weight[4] = -0.4;
    model.layer_d[1].weight[5] = -0.6;

    model.layer_d[1].biases[0] = -0.7;
    model.layer_d[1].biases[1] = -0.8;
}

// TODO: implement the feed-forward process to do inferencing.
void feedForward(const float *input, Model &model, float *output)
{
    model.layer_d[0].output = new float[model.layer_d[0].osize];
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[0].osize; i++) // 3
    {
        for (int j = 0; j < model.layer_d[0].isize; j++) // 2
        {
            model.layer_d[0].output[i] += input[j] * model.layer_d[0].weight[i * model.layer_d[0].isize + j];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[0].osize; i++)
    {
        model.layer_d[0].output[i] = model.layer_d[0].acts(model.layer_d[0].output[i] + model.layer_d[0].biases[i]);
    }
    model.layer_d[1].output = new float[model.layer_d[1].osize];
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[1].osize; i++) // 2
    {
        for (int j = 0; j < model.layer_d[1].isize; j++) // 3
        {
            model.layer_d[1].output[i] += model.layer_d[0].output[j] * model.layer_d[1].weight[i * model.layer_d[1].isize + j];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[1].osize; i++)
    {
        model.layer_d[1].output[i] = model.layer_d[1].acts(model.layer_d[1].output[i] + model.layer_d[1].biases[i]);
        output[i] = model.layer_d[1].output[i];
    }
}
// TODO: implement the back-propagate process to train your NN
float backPropagate(float *T, Model &model, float *X, float alpha)
{
    auto SSE = 0.0f;
    // alpha = 1;
    alpha = 0.1;
    float *EG = new float[model.layer_d[1].isize];
    model.layer_d[1].theta = new float[model.layer_d[1].osize];
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[1].osize; i++)
    {
        model.layer_d[1].theta[i] = (model.layer_d[1].output[i] - T[i]) * (model.layer_d[1].output[i] * (1 - model.layer_d[1].output[i]));
        SSE += pow((model.layer_d[1].output[i] - T[i]), 2) / 2;
    }
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[1].isize; i++) // 3
    {
        for (int j = 0; j < model.layer_d[1].osize; j++) // 2
        {
            EG[i] += model.layer_d[1].theta[j] * model.layer_d[1].weight[j * model.layer_d[1].isize + i];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[1].osize; i++) // 2
    {
        for (int j = 0; j < model.layer_d[1].isize; j++) // 3
        {
            model.layer_d[1].weight[i * model.layer_d[1].isize + j] = model.layer_d[1].weight[i * model.layer_d[1].isize + j] - (model.layer_d[1].theta[i] * model.layer_d[0].output[j]) * alpha;
        }
    }
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[1].osize; i++) // 2
    {
        model.layer_d[1].biases[i] = model.layer_d[1].biases[i] - model.layer_d[1].theta[i] * alpha;
    }
    model.layer_d[0].theta = new float[model.layer_d[0].osize];
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[0].osize; i++)
    {
        model.layer_d[0].theta[i] = EG[i] * (model.layer_d[0].output[i] * (1 - model.layer_d[0].output[i]));
    }
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[0].osize; i++) //
    {
        for (int j = 0; j < model.layer_d[0].isize; j++) //
        {
            model.layer_d[0].weight[i * model.layer_d[0].isize + j] = model.layer_d[0].weight[i * model.layer_d[0].isize + j] - (model.layer_d[0].theta[i] * X[j]) * alpha;
        }
    }
#pragma omp parallel for
    for (int i = 0; i < model.layer_d[0].osize; i++) //
    {
        model.layer_d[0].biases[i] = model.layer_d[0].biases[i] - model.layer_d[0].theta[i] * alpha;
    }
    return SSE;
}
