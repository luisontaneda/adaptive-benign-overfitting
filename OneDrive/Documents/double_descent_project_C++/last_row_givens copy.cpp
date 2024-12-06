#include <Eigen/Dense>
#include <iostream>
#include <lapacke.h>
#include "QR_RLS.h"

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
    double c, s;

    for (int i = 0; i < diag; i++)
    {
        double x = A(i, i);
        double y = A(A.rows() - 1, i);
        double r = std::sqrt(x * x + y * y);
        // double c = x / r;
        // double s = -y / r;
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

        G(i, i) = c;
        G(G.rows() - 1, G.cols() - 1) = c;
        G(i, G.cols() - 1) = -s;
        G(G.rows() - 1, i) = s;

        A = G * A;
        Q = Q * G.transpose();
        G = MatrixXld::Identity(A.rows(), A.rows());
    }

    all_Q = all_Q * Q;
    // A = Q.transpose() * A;

    qr_rls->A = A;
    qr_rls->all_Q = all_Q;
    // qr_rls->all_Q = all_Q.unaryExpr([&tolerance](double val)
    //                                 { return std::abs(val) < tolerance ? 0.0 : val; });

    // qr_rls->A = A.unaryExpr([&tolerance](double val)
    //                         { return std::abs(val) < tolerance ? 0.0 : val; });
    return {Q.transpose(), A};
}

MatrixXld givens_downdate(QR_Rls *qr_rls)
{
    MatrixXld P = qr_rls->P;
    MatrixXld all_Q = qr_rls->all_Q;
    MatrixXld A = qr_rls->A;

    MatrixXld G = MatrixXld::Identity(A.rows(), A.rows());
    MatrixXld G_all = MatrixXld::Identity(A.rows(), A.rows());
    double tolerance = 1e-14;

    // int diag = P.cols() - 1;
    MatrixXld q = all_Q.row(0).transpose(); // Reshape to a column vector
    // MatrixXld q = all_Q.col(0);

    std::cout << "pe" << std::endl;
    std::cout << q << std::endl;

    int n = 3;                               // Number of elements
    std::vector<double> x = {1.0, 4.0, 6.0}; // Vector x
    std::vector<double> y = {2.0, 3.0, 5.0}; // Vector y
    std::vector<double> c(n);                // Cosines
    std::vector<double> s(n);                // Sines

    LAPACK_dlargv(&n, x.data(), &incx, y.data(), &incy, c.data(), &incc);

    int diag = P.cols() - 1;
    //    MatrixXld q = all_Q.row(0).transpose(); // Reshape to a column vector
    double c;
    double s;

    for (int i = diag; i > 0; i--)
    // for (int i = 1; i < diag + 1; i++)
    {
        // Reset G to identity before each rotation
        G = MatrixXld::Identity(all_Q.rows(), all_Q.rows());

        // Use current element and the one to be zeroed
        double x = q(i - 1, 0); // Changed from q(0,0)
        double y = q(i, 0);

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

        // double r = std::sqrt(x * x + y * y);
        // double c = x / r;
        // double s = -y / r;

        G(i, i) = c;
        G(i - 1, i - 1) = c; // Changed from G(0,0)
        G(i - 1, i) = -s;    // Changed from G(0,i)
        G(i, i - 1) = s;     // Changed from G(i,0)

        // G(i, i) = c;
        // G(0, 0) = c;  // Changed from G(0,0)
        // G(i, 0) = s;  // Changed from G(0,i)
        // G(0, i) = -s; // Changed from G(i,0)

        // Apply rotations
        // A = G * A;
        q = G * q;
        G_all = G_all * G.transpose();
        // G_all = G_all * G;
        all_Q = all_Q * G.transpose();
    }

    cout << "je" << endl;
    cout << q << endl;

    // all_Q = all_Q * G_all;
    cout << "je" << endl;
    cout << A.col(0) << endl;
    cout << "je" << endl;
    cout << A.col(1) << endl;

    cout << "je" << endl;
    cout << all_Q.row(0) << endl;
    cout << "je" << endl;
    cout << all_Q.col(0) << endl;

    // qr_rls->all_Q = all_Q.unaryExpr([&tolerance](double val)
    //                                { return std::abs(val) < tolerance ? 0.0 : val; });
    qr_rls->all_Q = all_Q;
    // cout << "je" << endl;
    // cout << qr_rls->all_Q.row(0) << endl;
    // cout << "je" << endl;
    // cout << qr_rls->all_Q.col(0) << endl;
    //  qr_rls->all_Q;

    //::cout << qr_rls->all_Q << std::endl;
    // qr_rls->A = A;

    return G_all;
}