#include <Eigen/Dense>
#include <lapacke.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "QR_RLS.h"
#include <complex>

// extern "C"
//{
//     void dlartg(const double *a, const double *b, double *c, double *s, double *r);
//     void drot(const int *n, double *x, const int *incx, double *y, const int *incy, const double *c, const double *s);
// }

extern "C"
{
#include <cblas.h>
    void drotg_(double *a, double *b, double *c, double *s);
    void dlartg_(double *a, double *b, double *c, double *s, double *r);
    void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
               double *c, double *s);
}

double sign(double x)
{
    if (x > 0)
    {
        return 1.0;
    }
    else if (x < 0)
    {
        return -1.0;
    }

    return 0;
}

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXld;

std::pair<MatrixXld, MatrixXld> givens_update(QR_Rls *qr_rls)
{
    MatrixXld A = qr_rls->A;
    MatrixXld copy_A = qr_rls->A;
    MatrixXld G = MatrixXld::Identity(A.rows(), A.rows());
    MatrixXld all_Q = qr_rls->all_Q;
    all_Q.conservativeResize(all_Q.rows(), all_Q.cols() + 1);
    all_Q.conservativeResize(all_Q.rows() + 1, all_Q.cols());
    all_Q(all_Q.rows() - 1, all_Q.cols() - 1) = 1;
    double tolerance = 1e-14;
    int m = all_Q.rows();
    int n = all_Q.cols();

    MatrixXld Q = G;

    int diag = (A.rows() > A.cols()) ? A.cols() - 1 : A.rows() - 1;
    // double c, s, r;
    int inc = 1;

    // Generate Givens rotation parameters for x[0] and y[0]

    for (int i = 0; i < diag; i++)
    {
        double x = A(i, i);
        double y = A(A.rows() - 1, i);

        // drotg_(&x, &y, &c, &s);
        double c, s, r;
        // double x = all_Q(0, i - 1);
        // double y = all_Q(0, i);
        dlartg_(&x, &y, &c, &s, &r);

        // int remaining = n - i - 1;
        // if (remaining > 0)
        //{
        //  Apply rotation to row i of R and rest of w
        //    drot_(&remaining, &A(i, i), &m, &A(A.rows() - 1, i), &inc, &c, &s);
        //}

        // A(i, i) = r;
        // A(A.rows() - 1, i) = 0;
        cout << A.col(i) << endl;

        cout << "pepe" << endl;

        // Update A

        // drot_(&temp_n, &A(i, i), &inc, &A(m - 1, i), &inc, &c, &s);
        for (int j = 0; j < 1000; j++)
        {
            // int temp_n = A.cols() - j;
            int temp_n = 1;
            drot_(&temp_n, &A(i, j), &inc, &A(A.rows() - 1, j), &inc, &c, &s);
        }
        // for (int j = 0; j < A.cols(); ++j)
        //{
        //     int temp_n = n - j;
        //     drot_(&temp_n, &A(i, i), &inc, &A(m - 1, j), &inc, &c, &s);
        // }

        cout << A.col(i) << endl;

        // Update Q
        drot_(&m, &all_Q(i, i), &inc, &all_Q(m - 1, i), &inc, &c, &s);

        // drot_(&m, &Q(0, i), &inc, &Q(0, n - 1), &inc, &c, &s);
    }

    cout << (all_Q.transpose() * copy_A).col(7) << endl;
    cout << "je" << endl;

    cout << A.col(7) << endl;

    qr_rls->A = A;
    qr_rls->all_Q = all_Q;
    return {Q, A};
}

MatrixXld givens_downdate(QR_Rls *qr_rls)
{
    MatrixXld P = qr_rls->P;
    MatrixXld all_Q = qr_rls->all_Q;
    MatrixXld A = qr_rls->A;
    VectorXld z = qr_rls->z;
    int m = all_Q.rows();
    int n = all_Q.cols();
    int inc = 1;

    MatrixXld G = MatrixXld::Identity(A.rows(), A.rows());
    MatrixXld G_all = MatrixXld::Identity(A.rows(), A.rows());

    MatrixXld q = all_Q.row(0).transpose(); // Reshape to a column vector

    int diag = P.cols() - 1;

    for (int i = diag; i > 0; i--)
    {

        double c, s, r;
        double x = all_Q(0, i - 1);
        double y = all_Q(0, i);
        dlartg_(&x, &y, &c, &s, &r);

        // Update Q
        drot_(&m, &all_Q(0, i - 1), &inc, &all_Q(0, i), &inc, &c, &s);
    }

    qr_rls->all_Q = all_Q;
    cout << "we" << endl;
    cout << q << endl;
    //  cout << "je" << endl;
    cout << qr_rls->all_Q.row(0) << endl;
    cout << "je" << endl;
    cout << qr_rls->all_Q.col(0) << endl;
    // qr_rls->all_Q;

    //::cout << qr_rls->all_Q << std::endl;
    qr_rls->A = A;
    qr_rls->z = z;

    return G_all;
}