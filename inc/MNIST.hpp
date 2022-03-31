#pragma once
#include <cstdint>
#include <string>

// This struct stores labels from the MNIST dataset. 
struct MNISTLabels {
	int noItems;				// number of labels
	uint8_t* labels; 			// a 1-D array contains all the labels 
	float** codes; 				// a 1-D array contains labels encoded using one-hot encoding
};

// This struct stores images from the MINST dataset. 
struct MNISTImages {
	int noItems;				// number of images
	int rows, cols;				// number of rows and columns of each image (should be 28*28)
	uint8_t** images;			// an array of images, and each image has rows*columns pixel values
	float** normalizedImages; 	// an array of normalized images, and each normalize image's pixel values are between [0.0f, 1.0f]
};

// readLabels reads the MNIST labels and stores them into the labels struct from the MNIST dataset specified in the filename
bool readLabels(const std::string& filename, MNISTLabels& labels);

// readImages reads the MNIST images and stores them into the images struct from the MNIST dataset specified in the filename
bool readImages(const std::string& filename, MNISTImages& images); 
