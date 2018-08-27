#ifndef Image_hpp
#define Image_hpp
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "DenseMatrix.hpp"
#include "pca.hpp"
DenseMatrix<double> Mat2DenseMatrix(cv::Mat mat) {
    DenseMatrix<double> matrix(mat.rows, mat.cols, 0);
    for (int i = 0; i < matrix.Rows(); ++i) {
        for (int j = 0; j < matrix.Columns(); ++j) {
            matrix(i, j) = double(mat.at<uchar>(i, j));
        }
    }
    return matrix;
}
cv::Mat DenseMatrix2Mat(DenseMatrix<double> matrix) {
    cv::Mat mat = cv::Mat::zeros(matrix.Rows(), matrix.Columns(), CV_8U);
    for (int i = 0; i < matrix.Rows(); ++i) {
        for (int j = 0; j < matrix.Columns(); ++j) {
            if (matrix(i, j) >= 0) mat.at<uchar>(i, j) = (matrix(i, j));
        }
    }
    return mat;
}
cv::Mat readImage(std::string imageName) {
    cv::Mat image, image2;
    image = cv::imread(imageName, cv::IMREAD_COLOR);  // Read the file
    cv::cvtColor(image, image2, cv::COLOR_BGR2GRAY);
    return image2;
}

void saveRawImage(DenseMatrix<double> image, std::string filename) {
    std::ofstream fout;
    fout.open(filename);
    fout << image;
    fout.close();
}

DenseMatrix<double> ReadRawImage(std::string filename) {
    std::ifstream fin;
    fin.open(filename);
    DenseMatrix<double> image(0, 0, 0);
    fin >> image;
    return image;
}

void SaveCompressedImage(DenseMatrix<double> image, std::string filename,
                         int eigenVectors) {
    DenseMatrix<double> V = PCA(image);
    double n = image.Columns();
    DenseMatrix<double> summer(image.Columns(), 1, 1);
    DenseMatrix<double> mean = image * summer;
    mean = mean / n;
    std::cout << image.Columns() << std::endl;
    image = image - mean * Transpose(summer);
    DenseMatrix<double> V_ =
        V.GetSubMatrix(0, V.Columns() - eigenVectors, V.Rows(), eigenVectors);
    DenseMatrix<double> newData = ((Transpose(V_)) * image);
    std::ofstream fout;
    fout.open(filename);
    fout << mean;
    fout << V_;
    fout << newData;
    fout.close();
}

DenseMatrix<double> ReadCompressedImage(std::string filename) {
    std::ifstream fin;
    fin.open(filename);
    DenseMatrix<double> V_(0, 0, 0);
    DenseMatrix<double> data(0, 0, 0);
    DenseMatrix<double> mean(0, 0, 0);
    fin >> mean;
    fin >> V_;
    fin >> data;
    data = V_ * data;
    DenseMatrix<double> summer(data.Columns(), 1, 1);
    data = data + mean * Transpose(summer);
    return data;
}

#endif