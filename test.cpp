#include <omp.h>
#include <chrono>
#include <iostream>
#include "Image.hpp"
using namespace std;

int main() {
    // cv::Mat image = readImage("lights.jpg");
    // cv::Mat image = readImage("office-365-logo-small-1.png");
    // DenseMatrix<double> data = Mat2DenseMatrix(image);
    // saveRawImage(data, "lights.jpg.raw");
    auto start = chrono::high_resolution_clock::now();
    // SaveCompressedImage(data, "lights.jpg.compressed.100.2", 100);
    auto data = ReadCompressedImage("lights.jpg.compressed.100.2");
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout << diff.count();

    // DenseMatrix<double> V = PCA(data);
    // int k = 50;
    // DenseMatrix<double> V_ = V.GetSubMatrix(0, V.Columns() - k, V.Rows(),
    // k); DenseMatrix<double> newData =
    //     ReadCompressedImage("lights.jpg.compressed.100");


    auto image = DenseMatrix2Mat(data);
    cv::namedWindow("Display window",
                    cv::WINDOW_AUTOSIZE);  // Create a window for

    cv::imshow("Display window", image);
    cv::waitKey(0);
    return 0;
}