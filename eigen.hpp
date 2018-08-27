#ifndef eigen_hpp
#define eigen_hpp


#include "DenseMatrix.hpp"

#include <algorithm>
#include <cmath>

template <class Real>
class Eigenvalue {
    int n;
    int issymmetric;
    DenseMatrix<Real> d;
    DenseMatrix<Real> e;
    DenseMatrix<Real> V;
    void tred2() {
#if defined(_OPENMP)
#pragma omp parallel
        {
#pragma omp for collapse(1)
            for (int j = 0; j < n; j++) d(j, 0) = V(n - 1, j);
        }
#else
        for (int j = 0; j < n; j++) d(j, 0) = V(n - 1, j);
#endif


        for (int i = n - 1; i > 0; i--) {
            Real scale = 0.0;
            Real h = 0.0;

            for (int k = 0; k < i; k++) scale = scale + std::abs(d(k, 0));

            if (scale == 0.0) {
                e(i, 0) = d(i - 1, 0);

                for (int j = 0; j < i; j++) {
                    d(j, 0) = V(i - 1, j);
                    V(i, j) = 0.0;
                    V(j, i) = 0.0;
                }
            } else {
                for (int k = 0; k < i; k++) {
                    d(k, 0) /= scale;
                    h += d(k, 0) * d(k, 0);
                }

                Real f = d(i - 1, 0);
                Real g = sqrt(h);
                if (f > 0) g = -g;
                e(i, 0) = scale * g;
                h = h - f * g;
                d(i - 1, 0) = f - g;
#if defined(_OPENMP)
#pragma omp parallel
                {
#pragma omp for collapse(1)
                    for (int j = 0; j < i; j++) e(j, 0) = 0.0;
                }

#else
                for (int j = 0; j < i; j++) e(j, 0) = 0.0;

#endif

                for (int j = 0; j < i; j++) {
                    f = d(j, 0);
                    V(j, i) = f;
                    g = e(j, 0) + V(j, j) * f;


                    for (int k = j + 1; k <= i - 1; k++) {
                        g += V(k, j) * d(k, 0);
                        e(k, 0) += V(k, j) * f;
                    }
                    e(j, 0) = g;
                }
                f = 0.0;


                for (int j = 0; j < i; j++) {
                    e(j, 0) /= h;
                    f += e(j, 0) * d(j, 0);
                }


                Real hh = f / (h + h);

#if defined(_OPENMP)
#pragma omp parallel
                {
#pragma omp for collapse(1)
                    for (int j = 0; j < i; j++) e(j, 0) -= hh * d(j, 0);
                }

#else
                for (int j = 0; j < i; j++) e(j, 0) -= hh * d(j, 0);

#endif


                for (int j = 0; j < i; j++) {
                    f = d(j, 0);
                    g = e(j, 0);

                    for (int k = j; k <= i - 1; k++)
                        V(k, j) -= (f * e(k, 0) + g * d(k, 0));
                    d(j, 0) = V(i - 1, j);
                    V(i, j) = 0.0;
                }
            }
            d(i, 0) = h;
        }

        for (int i = 0; i < n - 1; i++) {
            V(n - 1, i) = V(i, i);
            V(i, i) = 1.0;
            Real h = d(i + 1, 0);
            if (h != 0.0) {
#if defined(_OPENMP)
#pragma omp parallel
                {
#pragma omp for collapse(1)
                    for (int k = 0; k <= i; k++) d(k, 0) = V(k, i + 1) / h;
                }
#else
                for (int k = 0; k <= i; k++) d(k, 0) = V(k, i + 1) / h;

#endif


                for (int j = 0; j <= i; j++) {
                    Real g = 0.0;


                    for (int k = 0; k <= i; k++) g += V(k, i + 1) * V(k, j);

#if defined(_OPENMP)
#pragma omp parallel
                    {
#pragma omp for collapse(1)
                        for (int k = 0; k <= i; k++) V(k, j) -= g * d(k, 0);
                    }
#else
                    for (int k = 0; k <= i; k++) V(k, j) -= g * d(k, 0);
#endif
                }
            }

#if defined(_OPENMP)
#pragma omp parallel
            {
#pragma omp for collapse(1)
                for (int k = 0; k <= i; k++) V(k, i + 1) = 0.0;
            }
#else
            for (int k = 0; k <= i; k++) V(k, i + 1) = 0.0;
#endif
        }

#if defined(_OPENMP)
#pragma omp parallel
        {
#pragma omp for collapse(1)
            for (int j = 0; j < n; j++) {
                d(j, 0) = V(n - 1, j);
                V(n - 1, j) = 0.0;
            }
        }
#else
        for (int j = 0; j < n; j++) {
            d(j, 0) = V(n - 1, j);
            V(n - 1, j) = 0.0;
        }
#endif
        V(n - 1, n - 1) = 1.0;
        e(0, 0) = 0.0;
    }

    void tql2() {
        for (int i = 1; i < n; i++) e(i - 1, 0) = e(i, 0);
        e(n - 1, 0) = 0.0;
        Real f = 0.0;
        Real tst1 = 0.0;
        Real eps = pow(2.0, -52.0);
        for (int l = 0; l < n; l++) {
            tst1 = std::max(tst1, std::abs(d(l, 0)) + std::abs(e(l, 0)));
            int m = l;
            while (m < n) {
                if (std::abs(e(m, 0)) <= eps * tst1) {
                    break;
                }
                m++;
            }

            if (m > l) {
                int iter = 0;
                do {
                    iter = iter + 1;

                    Real g = d(l, 0);
                    Real p = (d(l + 1, 0) - g) / (2.0 * e(l, 0));
                    Real r = hypot(p, 1.0);
                    if (p < 0) r = -r;
                    d(l, 0) = e(l, 0) / (p + r);
                    d(l + 1, 0) = e(l, 0) * (p + r);
                    Real dl1 = d(l + 1, 0);
                    Real h = g - d(l, 0);
                    for (int i = l + 2; i < n; i++) d(i, 0) -= h;
                    f = f + h;

                    p = d(m, 0);
                    Real c = 1.0;
                    Real c2 = c;
                    Real c3 = c;
                    Real el1 = e(l + 1, 0);
                    Real s = 0.0;
                    Real s2 = 0.0;
                    for (int i = m - 1; i >= l; i--) {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e(i, 0);
                        h = c * p;
                        r = hypot(p, e(i, 0));
                        e(i + 1, 0) = s * r;
                        s = e(i, 0) / r;
                        c = p / r;
                        p = c * d(i, 0) - s * g;
                        d(i + 1, 0) = h + s * (c * g + s * d(i, 0));

                        for (int k = 0; k < n; k++) {
                            h = V(k, i + 1);
                            V(k, i + 1) = s * V(k, i) + c * h;
                            V(k, i) = c * V(k, i) - s * h;
                        }
                    }
                    p = -s * s2 * c3 * el1 * e(l, 0) / dl1;
                    e(l, 0) = s * p;
                    d(l, 0) = c * p;

                } while (abs(e(l, 0)) > eps * tst1);
            }
            d(l, 0) = d(l, 0) + f;
            e(l, 0) = 0.0;
        }

        for (int i = 0; i < n - 1; i++) {
            int k = i;
            Real p = d(i, 0);
            for (int j = i + 1; j < n; j++) {
                if (d(j, 0) < p) {
                    k = j;
                    p = d(j, 0);
                }
            }
            if (k != i) {
                d(k, 0) = d(i, 0);
                d(i, 0) = p;
                for (int j = 0; j < n; j++) {
                    p = V(j, i);
                    V(j, i) = V(j, k);
                    V(j, k) = p;
                }
            }
        }
    }

    Real cdivr, cdivi;
    void cdiv(Real xr, Real xi, Real yr, Real yi) {
        Real r, d;
        if (abs(yr) > abs(yi)) {
            r = yi / yr;
            d = yr + r * yi;
            cdivr = (xr + r * xi) / d;
            cdivi = (xi - r * xr) / d;
        } else {
            r = yr / yi;
            d = yi + r * yr;
            cdivr = (r * xr + xi) / d;
            cdivi = (r * xi - xr) / d;
        }
    }

  public:
    Eigenvalue(const DenseMatrix<Real> &A)
        : n(A.Rows()), V(n, n, 0), d(n, 1, 0), e(n, 1, 0) {
        issymmetric = 1;

        if (issymmetric) {
#if defined(_OPENMP)
#pragma omp parallel
            {
#pragma omp for collapse(2)
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) V(i, j) = A(i, j);
            }
#else
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) V(i, j) = A(i, j);
#endif

            tred2();
            tql2();
        }
    }

    DenseMatrix<Real> getV() { return V; }

    DenseMatrix<Real> getRealEigenvalues() { return d; }

    DenseMatrix<Real> getImagEigenvalues() { return e; }

    DenseMatrix<Real> getD() {
        DenseMatrix<Real> D = DenseMatrix<Real>(n, n);

#if defined(_OPENMP)
#pragma omp parallel
        {
#pragma omp for collapse(1)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) D(i, j) = 0.0;
                D(i, i) = d(i, 0);
                if (e(i, 0) > 0)
                    D(i, i + 1) = e(i, 0);
                else if (e(i, 0) < 0)
                    D(i, i - 1) = e(i, 0);
            }
        }
#else
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) D(i, j) = 0.0;
            D(i, i) = d(i, 0);
            if (e(i, 0) > 0)
                D(i, i + 1) = e(i, 0);
            else if (e(i, 0) < 0)
                D(i, i - 1) = e(i, 0);
        }
#endif

        return D;
    }
};
#endif