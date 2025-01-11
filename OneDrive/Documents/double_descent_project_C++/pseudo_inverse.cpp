#include <iostream>
#include <lapacke.h>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRowMajor;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixColMajor;

MatrixXld pinv(const Eigen::MatrixXd &A, double tolerance = 1e-16)
{
    // Sample matrix

    // it has to be a C array
    // LDA The leading dimension of array A
    // LDU The leading dimension of the array U

    const int int_m = A.rows();
    const int int_n = A.cols();
    double A_copy[int_m * int_n];

    for (int i = 0; i < int_m; i++)
    {
        for (int j = 0; j < int_n; j++)
        {
            A_copy[i * int_n + j] = A(i, j);
        }
    }

    // Dimensions
    lapack_int m = int_m, n = int_n, lda = n;
    lapack_int ldu = m, ldvt = n;
    lapack_int info;
    double s[n], u[ldu * m], vt[ldvt * n];

    bool row_vec = A.rows() == 1;
    bool col_vec = A.cols() == 1;

    if (row_vec || col_vec)
    {
        double norm = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'F', m, n, A_copy, n);

        int vec_size = std::max(A.rows(), A.cols());

        for (int i = 0; i < vec_size; i++)
        {
            A_copy[i] /= (norm * norm);
        }

        if (row_vec)
        {
            Eigen::Map<Eigen::MatrixXd> B(A_copy, vec_size, 1);
            return B;
        }
        else
        {
            Eigen::Map<Eigen::MatrixXd> B(A_copy, 1, vec_size);
            return B;
        }
    }

    double atol = 0.0;
    double rtol = std::max<double>(m, n) * std::numeric_limits<double>::epsilon();

    // changed LAPACK_COL_MAJOR to LAPACK_COL_MAJOR
    info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', m, n, A_copy, lda, s,
                          u, ldu, vt, ldvt);

    Eigen::MatrixXd U_mat(m, m);
    Eigen::MatrixXd VT_mat(n, n);

    for (int i = 0; i < int_n; ++i)
    {
        for (int j = 0; j < int_n; ++j)
        {
            VT_mat(i, j) = vt[i * int_n + j];
        }
    }

    for (int i = 0; i < int_m; ++i)
    {
        for (int j = 0; j < int_m; ++j)
        {
            U_mat(i, j) = u[i * int_m + j];
        }
    }

    double maxS = *std::max_element(s, s + m);
    double val = atol + maxS * rtol;

    int rank = 0;
    for (int i = 0; i < m; ++i)
    {
        if (s[i] > val)
            rank++;
    }

    Eigen::MatrixXd u_rank = U_mat.leftCols(rank);

    for (int i = 0; i < rank; ++i)
    {
        u_rank.col(i) /= s[i];
    }

    VT_mat = VT_mat.topRows(rank);
    Eigen::MatrixXd B = (u_rank * VT_mat).conjugate();

    return B.transpose();
}