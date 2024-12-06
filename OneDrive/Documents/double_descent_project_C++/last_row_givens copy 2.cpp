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
    MatrixXld G = MatrixXld::Identity(A.rows(), A.rows());
    MatrixXld all_Q = qr_rls->all_Q;
    all_Q.conservativeResize(all_Q.rows(), all_Q.cols() + 1);
    all_Q.conservativeResize(all_Q.rows() + 1, all_Q.cols());
    all_Q(all_Q.rows() - 1, all_Q.cols() - 1) = 1;
    double tolerance = 1e-14;

    MatrixXld Q = G;

    int diag = (A.rows() > A.cols()) ? A.cols() - 1 : A.rows() - 1;
    double c, s, r;

    // Generate Givens rotation parameters for x[0] and y[0]

    for (int i = 0; i < diag; i++)
    {
        double x = A(i, i);
        double y = A(A.rows() - 1, i);

        drotg_(&x, &y, &c, &s);
        // dlartg_(&x, &y, &c, &s, &r);
        // if (abs(y) >= abs(x))
        //{
        //   double t = -x / y;
        //    s = 1 / sqrt(1 + t * t);
        //    c = s * t;
        //}
        // else
        //{
        //    double t = -y / x;
        //    c = 1 / sqrt(1 + t * t);
        //    s = c * t;
        //}

        G = MatrixXld::Identity(A.rows(), A.rows());

        G(i, i) = c;
        G(A.rows() - 1, A.rows() - 1) = c;
        G(i, A.rows() - 1) = s;
        G(A.rows() - 1, i) = -s;

        A = G.transpose() * A;
        Q = Q * G;
        all_Q = all_Q * G;
        // Q = G * Q;
        // all_Q = G * all_Q;
    }

    qr_rls->A = A;
    qr_rls->all_Q = all_Q;
    // qr_rls->all_Q = all_Q.unaryExpr([&tolerance](double val)
    //                                 { return std::abs(val) < tolerance ? 0.0 : val; });

    // qr_rls->A = A.unaryExpr([&tolerance](double val)
    //                         { return std::abs(val) < tolerance ? 0.0 : val; });
    return {Q, A};
}

MatrixXld givens_downdate(QR_Rls *qr_rls)
{
    MatrixXld P = qr_rls->P;
    MatrixXld all_Q = qr_rls->all_Q;
    MatrixXld A = qr_rls->A;
    VectorXld z = qr_rls->z;

    MatrixXld G = MatrixXld::Identity(A.rows(), A.rows());
    MatrixXld G_all = MatrixXld::Identity(A.rows(), A.rows());

    MatrixXld q = all_Q.row(0).transpose(); // Reshape to a column vector

    int diag = P.cols() - 1;
    //    MatrixXld q = all_Q.row(0).transpose(); // Reshape to a column vector

    for (int i = diag; i > 0; i--)
    // for (int i = 1; i < diag + 1; i++)
    {
        double x = q(i - 1, 0);
        double y = q(i, 0);
        // Givens rotation parameters
        double c, s, r; // cos(theta), sin(theta)

        // Perform the Givens rotation to zero out y
        // drotg_(&x, &y, &c, &s);
        // dlartg_(&x, &y, &c, &s, &r);
        if (abs(y) >= abs(x))
        {
            double t = -x / y;
            s = 1 / sqrt(1 + t * t);
            c = s * t;
        }
        else
        {
            double t = -y / x;
            c = 1 / sqrt(1 + t * t);
            s = c * t;
        }

        // c = abs(x) / sqrt(pow(x, 2) + pow(y, 2));
        // s = sign(x) * y / sqrt(pow(x, 2) + pow(y, 2));

        G = MatrixXld::Identity(A.rows(), A.rows());

        G(i - 1, i - 1) = c;
        G(i, i) = c;
        G(i - 1, i) = s;  // Changed from G(0,i)
        G(i, i - 1) = -s; // Changed from G(i,0)

        // q = G.transpose() * q;
        q(i - 1, 0) = c * q(i - 1, 0) - s * q(i, 0);
        A = G.transpose() * A;
        z = G.transpose() * z;
        all_Q = all_Q * G;
        G_all = G_all * G;

        cout << "we" << endl;
        cout << q << endl;

        // q(i - 1, 0) = r;

        int juju = 8;
    }

    // qr_rls->all_Q = all_Q.unaryExpr([&tolerance](double val)
    //                                { return std::abs(val) < tolerance ? 0.0 : val; });
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