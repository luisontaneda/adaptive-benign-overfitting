#include <Eigen/Dense>
#include <lapacke.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "QR_RLS.h"
#include <complex>

extern "C"
{
#include <cblas.h>
    void drotg_(double *a, double *b, double *c, double *s);
    void dlartg_(double *a, double *b, double *c, double *s, double *r);
    void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
               double *c, double *s);
    void dswap_(int *n, double *dx, int *incx, double *dy, int *incy);
}

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXld;

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> givens_update(QR_Rls *qr_rls)
{
    MatrixXld A = qr_rls->A;
    MatrixXld all_Q = qr_rls->all_Q;
    MatrixXld G = MatrixXld::Identity(A.rows(), A.rows());

    all_Q.conservativeResize(all_Q.rows() + 1, all_Q.cols() + 1);

    all_Q.row(all_Q.rows() - 1).setZero();
    all_Q.col(all_Q.cols() - 1).setZero();
    all_Q(all_Q.rows() - 1, all_Q.cols() - 1) = 1;

    int m = A.rows();
    int n = A.cols();
    // double *flat_A = A.data();
    // double *flat_all_Q = all_Q.data();
    int A_size = A.size(); // Total number of elements
    // double *flat_A = new double[A_size];
    double flat_A[A_size];
    // Copy the matrix data into the flat array
    // std::memcpy(flat_A, A.data(), A_size * sizeof(double));

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            flat_A[j * m + i] = A(i, j);
        }
    }

    int all_Q_size = all_Q.size(); // Total number of elements
    double flat_all_Q[all_Q_size];

    int flat_G_size = G.size(); // Total number of elements
    double flat_G[flat_G_size];

    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < m; i++)
        {
            flat_all_Q[j * m + i] = all_Q(i, j);

            flat_G[j * m + i] = G(i, j);
        }
    }

    double c, s, r;
    int limit = std::min(m - 1, n);

    for (int j = 0; j < limit; ++j)
    {

        int row_stride = 1;
        int col_stride = m;

        dlartg_(&flat_A[j * row_stride + j * col_stride], &flat_A[(m - 1) * row_stride + j * col_stride], &c, &s, &r);

        flat_A[j * row_stride + j * col_stride] = r;
        flat_A[(m - 1) * row_stride + j * col_stride] = 0;

        int temp = n - j - 1;
        int idx_1 = j * row_stride + (j + 1) * col_stride;
        int idx_2 = (m - 1) * row_stride + (j + 1) * col_stride;
        double *ptr_1 = &flat_A[idx_1];
        double *ptr_2 = &flat_A[idx_2];

        drot_(&temp, &flat_A[idx_1], &col_stride,
              &flat_A[idx_2], &col_stride, &c, &s);

        idx_1 = j * col_stride;
        idx_2 = (m - 1) * col_stride;

        drot_(&m, &flat_all_Q[j * col_stride], &row_stride,
              &flat_all_Q[(m - 1) * col_stride], &row_stride, &c, &s);

        drot_(&m, &flat_G[j * col_stride], &row_stride,
              &flat_G[(m - 1) * col_stride], &row_stride, &c, &s);
    }

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            A(i, j) = flat_A[j * m + i];
        }
    }

    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < m; i++)
        {
            all_Q(i, j) = flat_all_Q[j * m + i];
            G(i, j) = flat_G[j * m + i];
        }
    }

    qr_rls->all_Q = all_Q;
    qr_rls->A = A;

    return {G, qr_rls->A};
}

MatrixXld givens_downdate(QR_Rls *qr_rls)
{
    MatrixXld P = qr_rls->P;
    MatrixXld all_Q = qr_rls->all_Q;
    MatrixXld A = qr_rls->A;
    VectorXld z = qr_rls->z;
    int m = all_Q.rows();
    int n = A.cols();
    int inc = 1;

    MatrixXld G = MatrixXld::Identity(A.rows(), A.rows());
    MatrixXld G_all = MatrixXld::Identity(A.rows(), A.rows());

    MatrixXld q = all_Q.row(0).transpose(); // Reshape to a column vector

    int diag = P.cols() - 1;
    int row_stride = 1;
    int col_stride = m;
    double c, s, r;

    int A_size = A.size(); // Total number of elements
    double flat_A[A_size];

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            flat_A[j * m + i] = A(i, j);
        }
    }

    int all_Q_size = all_Q.size(); // Total number of elements
    double flat_all_Q[all_Q_size];

    int flat_G_size = G.size(); // Total number of elements
    double flat_G[flat_G_size];

    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < m; i++)
        {
            flat_all_Q[j * m + i] = all_Q(i, j);

            flat_G[j * m + i] = G(i, j);
        }
    }

    for (int i = m - 1; i > 0; --i)
    {
        // Generate Givens rotation

        dlartg_(&flat_all_Q[(i - 1) * col_stride], &flat_all_Q[i * col_stride], &c, &s, &r);

        // Update W
        int idx_1 = (i - 1) * col_stride;
        int idx_2 = i * col_stride;
        // drot_(&m, &flat_all_Q[(i - 1) * col_stride], &row_stride,
        //       &flat_all_Q[i * col_stride], &row_stride, &c, &s);

        flat_all_Q[(i - 1) * col_stride] = r;
        flat_all_Q[i * col_stride] = 0;

        int temp = m - 1;
        drot_(&temp, &flat_all_Q[((i - 1) * col_stride) + 1], &row_stride,
              &flat_all_Q[(i * col_stride) + 1], &row_stride, &c, &s);

        temp = n - i - 1;
        // drot_(&temp, &flat_A[i + (i - 1) * col_stride], &col_stride,
        //       &flat_A[(i + 1) + i * col_stride], &col_stride, &c, &s);
        idx_1 = (i - 1) + (i - 1) * col_stride;
        idx_2 = i + (i - 1) * col_stride;

        // drot_(&temp, &flat_A[(i - 1) + i * col_stride], &col_stride,
        //       &flat_A[i + i * col_stride], &col_stride, &c, &s);

        // drot_(&temp, &flat_A[(i - 1) + (i - 1) * col_stride], &col_stride,
        //       &flat_A[i + (i - 1) * col_stride], &col_stride, &c, &s);

        drot_(&m, &flat_G[((i - 1) * col_stride)], &row_stride,
              &flat_G[(i * col_stride)], &row_stride, &c, &s);
    }

    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < m; i++)
        {
            if (flat_all_Q[0] > 0)
            {
                all_Q(i, j) = flat_all_Q[j * m + i] * 1;
                G(i, j) = flat_G[j * m + i] * 1;
            }
            else
            {
                all_Q(i, j) = flat_all_Q[j * m + i] * -1;
                G(i, j) = flat_G[j * m + i] * -1;
            }
        }
    }

    qr_rls->all_Q = all_Q;
    qr_rls->A = A;
    qr_rls->z = z;

    return G;
}