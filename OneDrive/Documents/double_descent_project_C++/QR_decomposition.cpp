#include <cmath>
#include <iostream>
#include <lapacke.h>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRowMajor;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixColMajor;

std::pair<MatrixXld, MatrixXld> Q_R_compute(const MatrixRowMajor &A)
// int Q_R_compute(const MatrixRowMajor &A)
{
    // Copy input matrix A since LAPACKE_dgeqrf modifies it
    const int m = A.rows();
    const int n = A.cols();
    double A_copy[m * n];

    std::copy(A.data(), A.data() + (m * n), A_copy);
    // TAU will contain the scalar factors of elementary reflectors
    double tau[std::min(m, n)];

    // Workspace query
    double work_query;
    int lwork = -1;

    // Perform QR factorization
    int info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n,
                                   A_copy, m, tau,
                                   &work_query, lwork);

    // Extract R (upper triangular part of A_copy)

    MatrixXld R_eig(m, n);

    for (int i = 0; i < m; i++)
    {
        for (int j = i; j < n; j++)
        { // Note: j starts from i
            // R[i * n + j] = A_copy[j * m + i];
            R_eig(i, j) = A_copy[j * m + i];
        }
    }

    // Generate Q matrix using ORGQR
    double *Q = A_copy;
    info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, m,
                          Q, m, tau);

    Eigen::Map<MatrixXld> Q_eig(Q, m, m);

    return {Q_eig, R_eig};
};