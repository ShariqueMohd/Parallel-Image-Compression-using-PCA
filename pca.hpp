#ifndef pca_hpp
#define pca_hpp
#include <iostream>
#include "DenseMatrix.hpp"
#include "eigen.hpp"

DenseMatrix<double> PCA(DenseMatrix<double> data) {
    double n = data.Columns();
    DenseMatrix<double> summer(data.Columns(), 1, 1);
    DenseMatrix<double> mean = data * summer;
    mean = mean / n;
    data = data / n;
    DenseMatrix<double> covarianceMatrix =
        n * (data * (Transpose(data))) - mean * Transpose(mean);
    // std::cout << covarianceMatrix << std::endl;
    auto eigenHandler = Eigenvalue<double>(covarianceMatrix);
    return eigenHandler.getV();
}
#endif