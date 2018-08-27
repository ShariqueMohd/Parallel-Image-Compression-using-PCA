#include "DenseMatrix.hpp"

#ifndef DenseMatrix_cpp
#define DenseMatrix_cpp
#include <omp.h>
#include <iomanip>
#include <random>
template <typename datatype>
DenseMatrix<datatype>::DenseMatrix(unsigned long rows, unsigned long columns,
                                   datatype value)
    : rows(rows), columns(columns), values(rows * columns, value) {}

template <typename datatype>
DenseMatrix<datatype>::DenseMatrix(unsigned long size, datatype value)
    : DenseMatrix(size, size, value) {}

template <typename datatype>
DenseMatrix<datatype>::DenseMatrix(
    const std::initializer_list<std::initializer_list<datatype>> &list) {
    this->rows = list.size();
    this->columns = 0;
    for (const std::initializer_list<datatype> &row : list) {
        this->columns = row.size();
        this->values.insert(this->values.end(), row.begin(), row.end());
    }
}

template <typename datatype>
datatype &DenseMatrix<datatype>::operator()(unsigned long row,
                                            unsigned long column) {
    unsigned long index = row * this->columns + column;
    return values[index];
}

template <typename datatype>
datatype DenseMatrix<datatype>::operator()(unsigned long row,
                                           unsigned long column) const {
    unsigned long index = row * this->columns + column;
    return values[index];
}

template <typename datatype>
template <typename datatype2>
DenseMatrix<datatype> &DenseMatrix<datatype>::operator=(
    const DenseMatrix<datatype2> &rhs) {
    this->rows = rhs.Rows();
    this->columns = rhs.Columns();
    this->values.resize(this->rows * this->columns);
    for (unsigned long row = 0; row < Rows(); ++row)
        for (unsigned long column = 0; column < Columns(); ++column)
            (*this)(row, column) = rhs(row, column);
    return (*this);
}

template <typename datatype>
std::vector<DenseMatrix<datatype>> DenseMatrix<datatype>::GetColumnVectors()
    const {
    std::vector<DenseMatrix<datatype>> columnVectors(
        Columns(), DenseMatrix<datatype>(Rows(), 1, 0));
    for (unsigned long row = 0; row < Rows(); ++row)
        for (unsigned long column = 0; column < Columns(); ++column)
            columnVectors[column](row, 0) = (*this)(row, column);
    return columnVectors;
}

template <typename datatype>
DenseMatrix<datatype> DenseMatrix<datatype>::GetSubMatrix(
    unsigned long rowStart, unsigned long columnStart, unsigned long rowLimit,
    unsigned long columnLimit) const {
    DenseMatrix<datatype> subMatrix(rowLimit, columnLimit, 0);
    for (unsigned long row = 0; row < rowLimit; ++row)
        for (unsigned long column = 0; column < columnLimit; ++column)
            subMatrix(row, column) =
                (*this)(rowStart + row, columnStart + column);
    return subMatrix;
}

template <typename datatype>
void DenseMatrix<datatype>::SetSubMatrix(const DenseMatrix<datatype> &subMatrix,
                                         unsigned long rowStart,
                                         unsigned long columnStart,
                                         unsigned long rowLimit,
                                         unsigned long columnLimit) {
    for (unsigned long row = 0; row < rowLimit; ++row)
        for (unsigned long column = 0; column < columnLimit; ++column)
            (*this)(rowStart + row, columnStart + column) =
                subMatrix(row, column);
}


template <typename datatype>
DenseMatrix<datatype> DenseMatrix<datatype>::Zeros(unsigned long rows,
                                                   unsigned long columns) {
    return DenseMatrix<datatype>(rows, columns, 0);
}
template <typename datatype>
DenseMatrix<datatype> DenseMatrix<datatype>::Zeros(unsigned long size) {
    return DenseMatrix<datatype>(size, 0);
}
template <typename datatype>
DenseMatrix<datatype> DenseMatrix<datatype>::Identity(unsigned long size) {
    auto I = DenseMatrix<datatype>(size, 0);
    for (unsigned long i = 0; i < size; ++i) I(i, i) = 1;
    return I;
}
template <typename datatype>
DenseMatrix<datatype> DenseMatrix<datatype>::Random(unsigned long rows,
                                                    unsigned long columns) {
    DenseMatrix<datatype> randomMatrix =
        DenseMatrix<datatype>(rows, columns, 0);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> distribution{5, 2};
    for (unsigned long row = 0; row < randomMatrix.Rows(); ++row)
        for (unsigned long column = 0; column < randomMatrix.Columns();
             ++column)
            randomMatrix(row, column) = distribution(gen);
    return randomMatrix;
}


template <typename datatype>
std::ostream &operator<<(std::ostream &cout, const DenseMatrix<datatype> &m) {
    cout << m.Rows() << " x " << m.Columns() << " Matrix\n";
    for (unsigned long row = 0; row < m.Rows(); ++row) {
        for (unsigned long column = 0; column < m.Columns(); ++column) {
            cout << "\t" << std::setprecision(5) << m(row, column);
        }
        cout << std::endl;
    }
    return cout;
}

template <typename datatype>
std::ofstream &operator<<(std::ofstream &fout, const DenseMatrix<datatype> &m) {
    fout << m.Rows() << " " << m.Columns() << "\n";
    for (unsigned long i = 0; i < m.Rows(); ++i) {
        for (unsigned long j = 0; j < m.Columns(); ++j) {
            fout << m(i, j) << " ";
        }
        fout << "\n";
    }
    return fout;
}

template <typename datatype>
std::ifstream &operator>>(std::ifstream &fin, DenseMatrix<datatype> &m) {
    unsigned long numrows, numcolumns;
    fin >> numrows >> numcolumns;
    m = DenseMatrix<datatype>(numrows, numcolumns, 0);
    for (unsigned long i = 0; i < m.Rows(); ++i) {
        for (unsigned long j = 0; j < m.Columns(); ++j) {
            fin >> m(i, j);
        }
    }
    return fin;
}

template <typename datatype>
DenseMatrix<datatype> operator+(const DenseMatrix<datatype> &op1,
                                const DenseMatrix<datatype> &op2) {
    if (!(op1.Rows() == op2.Rows() && op1.Columns() == op2.Columns()))
        throw std::logic_error("Matrix Dimensions must agree");
    DenseMatrix<datatype> copy(op1);

#if defined(_OPENMP)
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (unsigned long row = 0; row < copy.Rows(); ++row)
            for (unsigned long column = 0; column < copy.Columns(); ++column)
                copy(row, column) = copy(row, column) + op2(row, column);
    }
#else
    for (unsigned long row = 0; row < copy.Rows(); ++row)
        for (unsigned long column = 0; column < copy.Columns(); ++column)
            copy(row, column) = copy(row, column) + op2(row, column);
#endif

    return copy;
}

template <typename datatype>
DenseMatrix<datatype> operator-(const DenseMatrix<datatype> &op1,
                                const DenseMatrix<datatype> &op2) {
    if (!(op1.Rows() == op2.Rows() && op1.Columns() == op2.Columns()))
        throw std::logic_error("Matrix Dimensions must agree");
    DenseMatrix<datatype> copy(op1);
#if defined(_OPENMP)
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (unsigned long row = 0; row < copy.Rows(); ++row)
            for (unsigned long column = 0; column < copy.Columns(); ++column)
                copy(row, column) = copy(row, column) - op2(row, column);
    }
#else
    for (unsigned long row = 0; row < copy.Rows(); ++row)
        for (unsigned long column = 0; column < copy.Columns(); ++column)
            copy(row, column) = copy(row, column) - op2(row, column);
#endif
    return copy;
}


template <typename datatype>
DenseMatrix<datatype> operator*(const DenseMatrix<datatype> &op1,
                                const DenseMatrix<datatype> &op2) {
    if (!(op2.Rows() == op1.Columns()))
        throw std::logic_error("Matrix Dimensions must agree");
    DenseMatrix<datatype> result =
        DenseMatrix<datatype>::Zeros(op1.Rows(), op2.Columns());
#if defined(_OPENMP)
    DenseMatrix<datatype> top2 = Transpose(op2);
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (unsigned long row = 0; row < result.Rows(); ++row) {
            for (unsigned long column = 0; column < result.Columns();
                 ++column) {
                for (unsigned long i = 0; i < op1.Columns(); ++i) {
                    result(row, column) =
                        result(row, column) + op1(row, i) * top2(column, i);
                }
            }
        }
    }
#else
    for (unsigned long row = 0; row < result.Rows(); ++row)
        for (unsigned long column = 0; column < result.Columns(); ++column)
            for (unsigned long i = 0; i < op1.Columns(); ++i)
                result(row, column) =
                    result(row, column) + op1(row, i) * op2(i, column);
#endif

    return result;
}
template <typename datatype>
DenseMatrix<datatype> operator*(const datatype &op1,
                                const DenseMatrix<datatype> &op2) {
    DenseMatrix<datatype> copy(op2);
#if defined(_OPENMP)
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (unsigned long row = 0; row < copy.Rows(); ++row)
            for (unsigned long column = 0; column < copy.Columns(); ++column)
                copy(row, column) = op1 * copy(row, column);
    }
#else
    for (unsigned long row = 0; row < copy.Rows(); ++row)
        for (unsigned long column = 0; column < copy.Columns(); ++column)
            copy(row, column) = op1 * copy(row, column);
#endif

    return copy;
}

template <typename datatype>
DenseMatrix<datatype> operator*(const DenseMatrix<datatype> &op1,
                                const datatype &op2) {
    DenseMatrix<datatype> copy(op1);
#if defined(_OPENMP)
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (unsigned long row = 0; row < copy.Rows(); ++row)
            for (unsigned long column = 0; column < copy.Columns(); ++column)
                copy(row, column) = copy(row, column) * op2;
    }
#else
    for (unsigned long row = 0; row < copy.Rows(); ++row)
        for (unsigned long column = 0; column < copy.Columns(); ++column)
            copy(row, column) = copy(row, column) * op2;
#endif
    return copy;
}

template <typename datatype>
DenseMatrix<datatype> operator/(const DenseMatrix<datatype> &op1,
                                const datatype &op2) {
    DenseMatrix<datatype> copy(op1);
#if defined(_OPENMP)
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (unsigned long row = 0; row < copy.Rows(); ++row)
            for (unsigned long column = 0; column < copy.Columns(); ++column)
                copy(row, column) = copy(row, column) / op2;
    }
#else
    for (unsigned long row = 0; row < copy.Rows(); ++row)
        for (unsigned long column = 0; column < copy.Columns(); ++column)
            copy(row, column) = copy(row, column) / op2;
#endif

    return copy;
}

template <typename datatype>
DenseMatrix<datatype> Transpose(const DenseMatrix<datatype> &matrix) {
    DenseMatrix<datatype> matrixTranspose(matrix.Columns(), matrix.Rows(), 0);
    for (unsigned long row = 0; row < matrixTranspose.Rows(); ++row)
        for (unsigned long column = 0; column < matrixTranspose.Columns();
             ++column)
            matrixTranspose(row, column) = matrix(column, row);
    return matrixTranspose;
}

#endif