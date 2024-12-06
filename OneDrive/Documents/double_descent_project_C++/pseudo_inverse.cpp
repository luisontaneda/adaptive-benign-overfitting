#include <iostream>
#include <lapacke.h>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRowMajor;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixColMajor;

MatrixXld pinv(const MatrixRowMajor &A, double tolerance = 1e-16)
{
    // Sample matrix

    // it has to be a C array
    // LDA The leading dimension of array A
    // LDU The leading dimension of the array U

    const int int_m = A.rows();
    const int int_n = A.cols();
    double A_copy[int_m * int_n];

    std::copy(A.data(), A.data() + (int_m * int_n), A_copy);

    // Dimensions
    lapack_int m = int_m, n = int_n, lda = n;
    lapack_int ldu = m, ldvt = n;
    lapack_int info;
    double s[n], U[ldu * m], VT[ldvt * n];

    info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', m, n, A_copy, lda, s,
                          U, ldu, VT, ldvt);

    // This copys the array column wise
    Eigen::Map<MatrixColMajor> UT_mat(U, int_m, int_m);
    Eigen::Map<MatrixColMajor> V_mat(VT, int_n, int_n);

    Eigen::MatrixXd S_inv(int_m, int_n);
    S_inv.setZero();

    // std::cout << S_inv << std::endl;

    for (lapack_int i = 0; i < std::min(m, n); i++)
    {
        if (s[i] > tolerance)
        {
            S_inv(i, i) = 1.0 / s[i];
        }
        else
        {
            S_inv(i, i) = 0.0;
        }
    }

    // MatrixXld P = (V_mat * S_inv.transpose() * UT_mat).cast<long double>();
    MatrixXld P = (V_mat * S_inv.transpose() * UT_mat);

    return P;
}