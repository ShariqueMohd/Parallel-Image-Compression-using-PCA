#ifndef DenseMatrix_hpp
#define DenseMatrix_hpp
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <vector>
template <typename datatype>
class DenseMatrix {
    unsigned long rows, columns;
    std::vector<datatype> values;

  public:
    DenseMatrix(unsigned long rows, unsigned long columns, datatype value);
    DenseMatrix(unsigned long size, datatype value);
    DenseMatrix(
        const std::initializer_list<std::initializer_list<datatype>> &matrix);

    unsigned long Columns() const { return this->columns; }
    unsigned long Rows() const { return this->rows; }

    datatype &operator()(unsigned long row, unsigned long column);
    datatype operator()(unsigned long row, unsigned long column) const;

    template <typename datatype2>
    DenseMatrix<datatype> &operator=(const DenseMatrix<datatype2> &rhs);

    std::vector<DenseMatrix<datatype>> GetColumnVectors() const;
    DenseMatrix<datatype> GetSubMatrix(unsigned long rowStart,
                                       unsigned long columnStart,
                                       unsigned long rowLimit,
                                       unsigned long columnLimit) const;

    void SetSubMatrix(const DenseMatrix<datatype> &subMatrix,
                      unsigned long rowStart, unsigned long columnStart,
                      unsigned long rowLimit, unsigned long columnLimit);

    static DenseMatrix<datatype> Zeros(unsigned long rows,
                                       unsigned long columns);
    static DenseMatrix<datatype> Zeros(unsigned long rows);
    static DenseMatrix<datatype> Identity(unsigned long rows);
    static DenseMatrix<datatype> Random(unsigned long rows,
                                        unsigned long columns);
};

template <typename datatype>
std::ostream &operator<<(std::ostream &cout, const DenseMatrix<datatype> &m);

template <typename datatype>
std::ofstream &operator<<(std::ofstream &fout, const DenseMatrix<datatype> &m);

template <typename datatype>
std::ifstream &operator>>(std::ifstream &fin, DenseMatrix<datatype> &m);


template <typename datatype>
DenseMatrix<datatype> operator+(const DenseMatrix<datatype> &op1,
                                const DenseMatrix<datatype> &op2);


template <typename datatype>
DenseMatrix<datatype> operator-(const DenseMatrix<datatype> &op1,
                                const DenseMatrix<datatype> &op2);


template <typename datatype>
DenseMatrix<datatype> operator*(const DenseMatrix<datatype> &op1,
                                const DenseMatrix<datatype> &op2);

template <typename datatype>
DenseMatrix<datatype> operator*(const datatype &op1,
                                const DenseMatrix<datatype> &op2);

template <typename datatype>
DenseMatrix<datatype> operator*(const DenseMatrix<datatype> &op1,
                                const datatype &op2);

template <typename datatype>
DenseMatrix<datatype> operator/(const DenseMatrix<datatype> &op1,
                                const datatype &op2);

template <typename datatype>
DenseMatrix<datatype> Transpose(const DenseMatrix<datatype> &matrix);

#include "DenseMatrix.cpp"

#endif