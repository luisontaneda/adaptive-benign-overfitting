#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>
#include <random>
#include <string>
#include "QR_decomposition.cpp"
#include "pseudo_inverse.cpp"
#include "last_row_givens.cpp"
#include "QR_RLS.h"

using namespace std;

QR_Rls::QR_Rls(const MatrixXd &x, const VectorXd &y, int max_obs, double ff, double lambda)
    : X(x), y(y), max_obs(max_obs), l(lambda), dim(x.cols()), n_batch(x.rows())
{
    MatrixXd I = MatrixXd::Identity(dim, dim);
    this->ff = sqrt(ff);
    this->b = 1;

    // Forgetting factor matrix
    B = MatrixXd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i)
    {
        B(i, i) = pow(this->ff, dim - i - 1);
    }
    X = X * B;

    auto [Q, R] = Q_R_compute(X);
    // Q_R_compute(X);

    R_inv = pinv(R);
    w = R_inv * Q.transpose() * y;

    z = Q.transpose() * y;

    // A and P were used as R and R inverse
    A = R;

    cout << A.col(0) << endl;
    P = R_inv;
    max_obs = max_obs;
    this->all_Q = Q;
    i = 1;
}

// Update method
void QR_Rls::update(MatrixXd new_x, const MatrixXd &new_y)
{
    // X.conservativeResize(Eigen::NoChange, X.cols() + 1);
    X.conservativeResize(X.rows() + 1, Eigen::NoChange);
    X.bottomRows(new_x.rows()) = new_x;

    y.conservativeResize(new_y.rows() + y.rows(), 1);
    y.bottomRows(new_y.rows()) = new_y;

    z.conservativeResize(new_y.rows() + z.rows(), 1);
    z.bottomRows(new_y.rows()) = new_y;

    int nobs = X.rows();
    P = (1.0 / ff) * P;
    A = ff * A;

    RowVectorXd d = new_x * P;
    MatrixXd c = new_x * (MatrixXd::Identity(A.cols(), A.cols()) - P * A);

    // Update for new regime
    if (!(c.isApprox(RowVectorXd::Zero(c.cols()))))
    {
        MatrixXd c_inv = pinv(c);
        P.leftCols(P.cols()) = P - c_inv * d;
        P.conservativeResize(Eigen::NoChange, P.cols() + 1);
        P.col(P.cols() - 1) = c_inv; // Assuming last column is an identity
    }
    // Update for old regime
    else
    {
        VectorXd b_k = (1.0 / (1.0 + d.dot(d))) * P * d.transpose();
        P = MatrixXd::Zero(P.rows(), P.cols() + 1);
        P.leftCols(P.cols() - 1) = P - b_k * d.transpose();
        P.col(P.cols() - 1) = b_k; // Assuming last column is an identity
    }

    A.conservativeResize(A.rows() + 1, A.cols());
    A.row(A.rows() - 1) = new_x;

    auto [Q, A] = givens_update(this); // Assuming givens is implemented

    // cout << z << endl;

    // this->w = P * z;
    this->w = P * all_Q * y;

    auto [Q_test, R_test] = Q_R_compute(X);

    cout << R_test.col(2) << endl;

    cout << "je " << endl;

    cout << A.col(2) << endl;
    // Q_R_compute(X);

    VectorXd w_test = pinv(R_test) * Q_test.transpose() * y;

    cout << "real W" << w_test << endl;

    cout << "fake W" << w << endl;

    double mse_test = 0;

    for (int i = 0; i < w.size(); i++)
    {
        mse_test = pow((w_test(i) - this->w(i)), 2);
    }

    // this->w = P * y;
    this->P = P * Q;
    this->z = Q * z;
    i++;

    if (nobs > max_obs)
    {

        // cout << w << endl;
        //  VectorXld x_first = X.col(0);
        downdate();
        // cout << "hola" << endl;
    }
}

// void QR_Rls::downdate(MatrixXd new_x, const MatrixXd &new_y)
void QR_Rls::downdate()
{
    // Check the condition
    bool temp = (A.transpose() * P.transpose()).isApprox(MatrixXd::Identity(A.cols(), A.cols()));

    // Update matrices

    Q = givens_downdate(this);
    P = P * Q;
    A = Q.transpose() * A;

    // cout << A.col(0) << endl;

    RowVectorXd x_T = A.row(0);
    VectorXd c = MatrixXd::Zero(P.cols(), 1);
    c(0, 0) = 1.0;

    // Deletion for new regime
    if (!temp)
    {
        MatrixXd k = P * c;
        MatrixXd h = x_T * P;

        P = P - k * pinv(k) * P - P * pinv(h) * h + (pinv(k) * P * pinv(h))(0) * k * h;
    }
    // Deletion for old regime
    else
    {
        VectorXd x_neg_T = -x_T;
        VectorXd h = x_neg_T * P;
        VectorXd u = (MatrixXld::Identity(P.cols(), P.cols()) - A * P) * c;
        VectorXd h_T = h.transpose();
        VectorXd u_T = u.transpose();

        VectorXd k = P * c;
        double h_mag = h.dot(h_T);
        double u_mag = u_T.dot(u);
        double S = 1.0 + (x_neg_T * P * c)(0);
        MatrixXd p_2 = -((u_mag) / S * P * h_T) - k;
        MatrixXd q_2 = -((h_mag) / S * u.transpose() - h);
        double sigma_2 = h_mag * u_mag + S * S;
        P = P + (1 / S) * P * h.transpose() * u.transpose() - (S / sigma_2) * p_2 * q_2;
    }

    double tolerance = 1e-13;

    // P = P.unaryExpr([&tolerance](double val)
    //                 { return std::abs(val) < tolerance ? 0.0 : val; });

    cout << P.col(0) << endl;

    A = A.bottomRows(A.rows() - 1);

    z = all_Q.transpose() * y;
    // z = Q.transpose() * z;
    P = P.rightCols(P.cols() - 1);
    z = z.bottomRows(z.rows() - 1);
    y = y.bottomRows(y.rows() - 1);
    X = X.bottomRows(X.rows() - 1);
    cout << "ya nuwvo" << endl;
    cout << z << endl;
    w = P * z;

    auto [Q_test, R_test] = Q_R_compute(X);
    // Q_R_compute(X);

    VectorXd w_test = pinv(R_test) * Q_test.transpose() * y;

    cout << "real W" << w_test << endl;

    cout << "fake W" << w << endl;

    double mse_test = 0;

    for (int i = 0; i < w.size(); i++)
    {
        mse_test = pow((w_test(i) - w(i)), 2);
    }
    mse_test = mse_test / w.size();

    all_Q = all_Q.bottomRows(all_Q.rows() - 1);
    all_Q = all_Q.rightCols(all_Q.cols() - 1);
    // P = P.rightCols(P.cols() - 1);
    // z = z.bottomRows(z.rows() - 1);
    // y = y.bottomRows(y.rows() - 1);
}

double QR_Rls::pred(const MatrixXd &x)
{
    double pred_value;
    // cout << w << endl;
    if (x.rows() == 1)
    {
        pred_value = (x * w)(0, 0);
    }
    // else
    //{
    //     Eigen::MatrixXd pred_matrix = x.transpose() * w;
    //     pred_value = x.transpose() * self.w;
    // }
    return pred_value;
}
