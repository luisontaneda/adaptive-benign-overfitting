#include <Eigen/Dense>
#include <iostream>

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

    MatrixXld Q = G;

    int diag = (A.rows() > A.cols()) ? A.cols() - 1 : A.rows() - 1;

    for (int i = 0; i < diag; i++)
    {
        double x = A(i, i);
        double y = A(A.rows() - 1, i);
        double r = std::sqrt(x * x + y * y);
        double c = x / r;
        double s = -y / r;

        G(i, i) = c;
        G(G.rows() - 1, G.cols() - 1) = c;
        G(i, G.cols() - 1) = -s;
        G(G.rows() - 1, i) = s;

        A = G * A;
        Q = Q * G.transpose();
        G = MatrixXld::Identity(A.rows(), A.rows());
    }

    qr_rls->all_Q = all_Q * Q;
    return {Q.transpose(), A};
}

MatrixXld givens_downdate(QR_Rls *qr_rls)
{
    MatrixXld P = qr_rls->P;
    MatrixXld all_Q = qr_rls->all_Q;
    MatrixXld A = qr_rls->A;

    MatrixXld G = MatrixXld::Identity(P.cols(), P.cols());
    MatrixXld G_all = G;

    int diag = P.cols() - 1;
    MatrixXld q = all_Q.row(0).transpose(); // Reshape to a column vector

    for (int i = diag; i > 0; i--)
    {
        double x = q(0, 0);
        double y = q(i, 0);
        double r = std::sqrt(x * x + y * y);
        double c = x / r;
        double s = -y / r;

        G(i, i) = c;
        G(0, 0) = c;
        G(0, i) = -s;
        G(i, 0) = s;

        A = G * A;
        G_all = G_all * G.transpose();
        q = G * q;
        G = MatrixXld::Identity(all_Q.rows(), all_Q.rows());
    }

    all_Q = all_Q * G_all;
    return G_all;
}