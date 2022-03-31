#include <iostream>
#include <string>
#include <algorithm>
#include <tuple>
#include "../inc/stopwatch.hpp"
using namespace std;

#include "../inc/ANN.hpp"

// This function reads the entire MNIST dataset.
bool readData(const string &labelFile, const string &imageFile, MNISTLabels &labelData, MNISTImages &imageData)
{
    if (!readLabels(labelFile, labelData))
    {
        cerr << "\nError reading labels: " << labelFile << endl;
        return false;
    }
    if (!readImages(imageFile, imageData))
    {
        cerr << "\nError reading images: " << imageFile << endl;
        return false;
    }
    return true;
}

// This function uses structured-binding (since C++17) to return labels & images
tuple<MNISTLabels, MNISTImages> testSet()
{
    MNISTLabels labels;
    MNISTImages images;
    string label = "MNIST/t10k-labels.idx1-ubyte";
    string image = "MNIST/t10k-images.idx3-ubyte";
    if (!readData(label, image, labels, images))
    {
        cerr << "Error reading testing data." << endl;
    }

    return {labels, images};
}

tuple<MNISTLabels, MNISTImages> trainSet()
{
    MNISTLabels labels;
    MNISTImages images;
    string label = "MNIST/train-labels.idx1-ubyte";
    string image = "MNIST/train-images.idx3-ubyte";
    if (!readData(label, image, labels, images))
    {
        cerr << "Error reading training data." << endl;
    }
    return {labels, images};
}

void MNIST()
{
    auto [labels, images] = trainSet();
    auto [tstLbls, tstImgs] = testSet();

    // Do data pre-processing
    oneHotEncoding(labels);
    normalizeImages(images);
    normalizeImages(tstImgs);

    // Build ANN model.  In MNIST, each images has 28*28=784 pixels, so the input layer has 784 neurons.
    // 100 nodes/neurons for the hidden layer,
    // 10 nodes/neurons for the output layer because we are using one-hot encoding with 10 classes/labels.
    int sizes[] = {784, 100, 10};
    ActivationFunction acts[] = {sigmoid, sigmoid}; // three layers of neurons has two activation functions.
    auto model = buildModel(3, sizes, acts);

    initializeModel(model);

    // Train the network for 10 epoches/times
    float out[10];
    for (auto epoch = 1; epoch <= 10; ++epoch)
    {
        auto r = 0.0f;
        for (auto no = 0; no < images.noItems; ++no)
        {
            auto x = images.normalizedImages[no];
            feedForward(x, model, out);
            r += backPropagate(labels.codes[no], model, x);
        }
        cout << "\n"
             << epoch << ":" << r << " ";

        // testing
        auto correct = 0;
        for (auto no = 0; no < tstImgs.noItems; ++no)
        {
            auto x = tstImgs.normalizedImages[no];
            feedForward(x, model, out);
            auto which = std::max_element(out, out + 10) - out;
            if (which == tstLbls.labels[no])
                correct++;
        }
        cout << ", accuracy: " << 100.0f * correct / tstImgs.noItems << "%    ";
    }

    destroyModel(model);
    // del(labels, images, tstLbls, tstImgs);
}

// You can use this to verify your implementation by comparing with calculations in the given Excel.
void verify()
{
    float x[] = {1.0f, 2.0f};   // input
    float T[] = {0.25f, 0.75f}; // target
    float o[2] = {0.0f, 0.0f};  // output

    // Build ANN model
    int sizes[] = {2, 3, 2};
    ActivationFunction acts[] = {sigmoid, sigmoid}; // an array of function pointers
    auto model = buildModel(3, sizes, acts);
    initializeModel(model);
    testWeights(model);

    for (auto i = 0; i < 5; ++i)
    {
        feedForward(x, model, o);
        auto SSE = backPropagate(T, model, x, 1.0f);
        cout << "\n"
             << i << ":" << SSE << "[ " << o[0] << ", " << o[1] << " ]" << flush;
    }
    destroyModel(model);
}

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        cerr << argv[0] << " verify: do the simple NN to verify the implementation." << endl;
        cerr << argv[0] << " mnist: do the MNIST dataset." << endl;
        return 255;
    }
    auto action = string(argv[1]);
    if (action == "verify")
    {
        verify();
        return 0;
    }
    if (action == "mnist")
    {
        stopwatch t[1];
        t[0].start();
        MNIST();
        t[0].stop();
        cout << "\ntime:" << t[0].elapsedTime() << endl;
        return 0;
    }
    return 254;
}
