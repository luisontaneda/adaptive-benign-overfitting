#include <cmath>
#include <iostream>
#include <lapacke.h>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRowMajor;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixColMajor;

std::pair<MatrixXld, MatrixXld> Q_R_compute(const MatrixXld &A)
{
    // Dimensions of the matrix
    const int m = A.rows();
    const int n = A.cols();

    // Copy input matrix to column-major format for LAPACK
    std::vector<double> A_copy(m * n);
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            A_copy[j * m + i] = A(i, j);
        }
    }

    // Allocate space for tau
    std::vector<double> tau(std::min(m, n));

    // Workspace query
    double work_query;
    int lwork = -1;

    int info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n,
                                   A_copy.data(), m, tau.data(),
                                   &work_query, lwork);
    if (info != 0)
    {
        throw std::runtime_error("LAPACKE_dgeqrf_work failed during workspace query");
    }

    // Allocate workspace
    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);

    // Perform QR factorization
    info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n,
                               A_copy.data(), m, tau.data(),
                               work.data(), lwork);
    if (info != 0)
    {
        throw std::runtime_error("LAPACKE_dgeqrf_work failed");
    }

    // Extract R (upper triangular part)
    MatrixXld R_eig = MatrixXld::Zero(m, n);
    for (int i = 0; i < m; i++)
    {
        for (int j = i; j < n; j++)
        {
            R_eig(i, j) = A_copy[j * m + i];
        }
    }

    // Generate Q matrix using ORGQR
    info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, m, std::min(m, n),
                          A_copy.data(), m, tau.data());
    if (info != 0)
    {
        throw std::runtime_error("LAPACKE_dorgqr failed");
    }

    // Copy Q into an Eigen matrix
    MatrixXld Q_eig(m, m);
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < m; i++)
        {
            Q_eig(i, j) = A_copy[j * m + i];
        }
    }

    return {Q_eig, R_eig};
}