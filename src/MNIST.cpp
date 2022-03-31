#include "MNIST.hpp"

#include <fstream>
#include <iostream>
using namespace std;

// the raw data area stored in Big-endian (https://en.wikipedia.org/wiki/Endianness) order,
// but Intel processors use little-endian format in-memory, so we cannot directly
// read the file content and stored it into memory...
static int32_t readInt(istream &inp)
{
    unsigned char buf[4];
    inp.read((char *)buf, 4);
    return buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];
}

// Details: http://yann.lecun.com/exdb/mnist/
bool readLabels(const string &filename, MNISTLabels &labels)
{
    ifstream inp(filename, ios_base::in | ios_base::binary);
    if (!inp)
        return false;
    auto magic = readInt(inp);
    if (magic != 2049)
        return false;
    auto noLabels = readInt(inp); // This is the number of labels
    labels.noItems = int(noLabels);
    labels.labels = new uint8_t[labels.noItems];
    // TODO: You should record something and allocate enough memory to store incoming labels

    for (auto i = 0; i < noLabels; ++i)
    {
        char label;
        inp.read(&label, 1);
        labels.labels[i] = uint8_t(label);
        // TODO: store the label into proper place in the labels struct.
    }
    inp.close();
    return true;
}

bool readImages(const string &filename, MNISTImages &images)
{
    ifstream inp(filename, ios_base::in | ios_base::binary);
    auto magic = readInt(inp);
    if (magic != 2051)
        return false;
    auto noImages = readInt(inp); // This is the number of images
    auto rows = readInt(inp);     // number of rows of each image
    auto cols = readInt(inp);     // number of columns of each image

    // cout << "\nMaintain your images struct @ " << __LINE__ << ", " << __FILE__;
    // Now we know how many images are there, and how big (rows*cols) each image is.
    // TODO: You should store these informations into your struct, and allocate enough memory to store all the images
    images.noItems = int(noImages);
    images.rows = int(rows);
    images.cols = int(cols);
    auto anImage = new char[images.rows * images.cols];
    images.images = new uint8_t *[images.noItems];

    for (auto i = 0; i < images.noItems; ++i)
    {
        images.images[i] = new uint8_t[images.rows * images.cols];
        inp.read(anImage, rows * cols);
#pragma omp parallel for
        for (int j = 0; j < images.rows * images.cols; j++)
        {
            images.images[i][j] = uint8_t(anImage[j]);
        }
    }
    delete[] anImage;
    inp.close();
    return true;
}
