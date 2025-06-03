// This c++ code is an adapted version from sedpreps dsh_basis function
//
// Copyright (C) 2020 Helmholtz Centre Potsdam
// GFZ German Research Centre for Geosciences, Potsdam, Germany
// (https://www.gfz-potsdam.de)
//
// sedprep is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.


#include <cmath>
#include <iostream>
#include <pybind11/numpy.h>

namespace py = pybind11;

int lm2i(int l, int m)
{
    int i = l*l - 1 + 2 * std::abs(m);
    if(0<m){
        return i-1;
    } else {
        return i;
    }
}

py::array_t<double> dspharm(int lmax, py::array_t<double> z, double R)
{
    const int N = lmax*(lmax+2);
    int I = z.shape(1);

    auto result = py::array_t<double>(N * 3*I);

    py::buffer_info zbuf = z.request();
    double *zptr = static_cast<double *>(zbuf.ptr);

    py::buffer_info rbuf = result.request();
    double *rptr = static_cast<double *>(rbuf.ptr);

    for (int i=0; i < N * 3*I; i++) {
        rptr[i] = 0;
    }
    double sqrt2 = std::sqrt(2.);

    #pragma omp parallel for 
    for (int i=0; i < I; i++) {
        auto cos_t = std::cos(M_PI*zptr[i*3]/180);
        int ind = 0;

        for (int l=0; l <= lmax; l++) {
            for (int m=0; m <= l; m++) {
                double Plm = std::assoc_legendre(l, m, cos_t);
                double Plm2 = Plm / 2.;
                // North Component
                // The following recurrence formula is used:
                // d_theta P_l^m = -1/2[(l+m)(l-m+1)P_l^{m-1} - P_l^{m+1}]

                // P_l^{-1} term is treated separately
                if (0 < l && m == 1) {
                    ind = lm2i(l, 0);       // According order is zero
                    // The pre-fractors of the recurrence formula and the
                    // scaling for negative orders cancel out. The plus sign
                    // accounts for the Condon-Shortley phase
                    rptr[ind*I*3 + i*3] += Plm / 2;
                }
                // P_l^{m-1} term: m -> m+1
                if (0 < l && m < l) {       // Skip l=0; m=0, ..., l-1
                    ind = lm2i(l, m+1);
                    rptr[ind*I*3 + i*3] -= (l+m+1)*(l-m)*Plm2;
                    // Negative orders
                    rptr[(ind+1)*I*3 + i*3] -= (l+m+1)*(l-m)*Plm2;
                }
                // P_l^{m+1} term: m -> m-1
                if (0 < l && 0 < m) {       // Skip l=0; m=1, ..., l
                    ind = lm2i(l, m-1);
                    rptr[ind*I*3 + i*3] += Plm2;
                    // Negative orders
                    if (m-1 != 0) {         // Do not visit m=0 twice
                        rptr[(ind+1)*I*3 + i*3] += Plm2;
                    }
                }

                // East Component
                // The following recurrence formulas is used:
                // P_l^m/sin(theta)
                //       = -1/(2m)[P_{l-1}^{m-1} + (l+m-1)(l+m)P_{l-1}^{m-1}]
                // Does not visit m=0 at all since it is zero anyway
                // P_{l-1}^{m+1} term: l -> l+1 and m -> m-1
                if (l < lmax && 1 < m) {    // Skip l=lmax; m=2, ..., l
                    ind = lm2i(l+1, m-1);
                    rptr[ind*I*3 + i*3+1] -= Plm2/(m-1);
                    rptr[(ind+1)*I*3 + i*3+1] -= Plm2/(m-1);
                }
                // P_{l-1}^{m-1} term: l -> l+1 and m -> m+1
                if (l < lmax) {             // Skip l=lmax; m=0, ..., l
                    ind = lm2i(l+1, m+1);
                    rptr[ind*I*3 + i*3+1] -= (l+m+1)*(l+m+2)*Plm2/(m+1);
                    rptr[(ind+1)*I*3 + i*3+1] -= (l+m+1)*(l+m+2)*Plm2/(m+1);
                }

                // Down Component
                if (0 < l) {                // Skip l=0; m=0, ..., l
                    ind = lm2i(l, m);
                    // Zero and positive orders
                    rptr[ind*I*3 + i*3+2] = Plm;
                    // Negative orders
                    if (m != 0) {           // Do not visit m=0 twice
                        rptr[(ind+1)*I*3 + i*3+2] = Plm;
                    }
                }

            }
        }

        for (int l=1; l <= lmax; l++) {
            for (int m=0; m <= l; m++) {
                ind = lm2i(l, m);
                double m_deg_p = M_PI*(m*zptr[i*3+1])/180.;
                double sin_p = std::sin(m_deg_p);
                double cos_p = std::cos(m_deg_p);

                /* Condon-Shortley phase is somehow disagreeing with reference,
                   although the c++-reference says it's not included...
                   This also leads to a minus sign appearing further below.
                if (m % 2 != 0) {
                    rptr[ind*I*3 + i*3] *= -1.;
                    rptr[(ind+1)*I*3 + i*3] *= -1.;
                    rptr[ind*I*3 + i*3+1] *= -1.;
                    rptr[(ind+1)*I*3 + i*3+1] *= -1.;
                    rptr[ind*I*3 + i*3+2] *= -1.;
                    rptr[(ind+1)*I*3 + i*3+2] *= -1.;
                }
                */

                // North component
                rptr[ind*I*3 + i*3] *= -cos_p;
                // East component
                rptr[ind*I*3 + i*3+1] *= -m*sin_p;
                // Down component
                rptr[ind*I*3 + i*3+2] *= -(l+1)*cos_p;

                // Scaling
                double scl = std::pow((R / (zptr[i*3+2]*1.)), l+2);

                rptr[ind*I*3 + i*3] *= scl;
                rptr[ind*I*3 + i*3+1] *= scl;
                rptr[ind*I*3 + i*3+2] *= scl;

                if (m != 0) {
                    // Account for real form
                    rptr[ind*I*3 + i*3] *= sqrt2;
                    rptr[(ind+1)*I*3 + i*3] *= sqrt2;
                    rptr[ind*I*3 + i*3+1] *= sqrt2;
                    rptr[(ind+1)*I*3 + i*3+1] *= sqrt2;
                    rptr[ind*I*3 + i*3+2] *= sqrt2;
                    rptr[(ind+1)*I*3 + i*3+2] *= sqrt2;

                    // negative orders
                    // North component
                    rptr[(ind+1)*I*3 + i*3] *= -sin_p;
                    // East component
                    rptr[(ind+1)*I*3 + i*3+1] *= m*cos_p;
                    // Down component
                    rptr[(ind+1)*I*3 + i*3+2] *= -(l+1)*sin_p;

                    // Scaling
                    rptr[(ind+1)*I*3 + i*3] *= scl;
                    rptr[(ind+1)*I*3 + i*3+1] *= scl;
                    rptr[(ind+1)*I*3 + i*3+2] *= scl;

                    // Schmidt semi-norm
                    double norm = std::exp((std::lgamma(l-std::abs(m)+1)
                                            - std::lgamma(l+std::abs(m)+1))
                                           / 2);

                    rptr[ind*I*3 + i*3] *= norm;
                    rptr[(ind+1)*I*3 + i*3] *= norm;
                    rptr[ind*I*3 + i*3+1] *= norm;
                    rptr[(ind+1)*I*3 + i*3+1] *= norm;
                    rptr[ind*I*3 + i*3+2] *= norm;
                    rptr[(ind+1)*I*3 + i*3+2] *= norm;
                }
            }   // mloop
        }   // lloop
    }   // iloop

    result.resize({N, I*3});
    return result;
}
