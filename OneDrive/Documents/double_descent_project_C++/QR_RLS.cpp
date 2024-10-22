#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>
#include <random>
#include <string>
#include "QR_RLS.h"
#include "givens_rotations.cpp"
#include "pseudo_inverse.cpp"
#include "last_row_givens.cpp"

using namespace std;

QR_Rls::QR_Rls(const MatrixXld &x, const VectorXld &y, int max_obs, double ff, double lambda)
    : X(x), y(y), max_obs(max_obs), l(lambda), dim(x.rows()), n_batch(x.cols())
{
    MatrixXld I = MatrixXld::Identity(dim, dim);
    this->ff = sqrt(ff);
    this->b = 1;

    // Forgetting factor matrix
    B = MatrixXld::Zero(dim, dim);
    for (int i = 0; i < dim; ++i)
    {
        B(i, i) = pow(this->ff, dim - i - 1);
    }
    X = B * X;

    auto [Q, R] = Givens_Rotation(X);

    R_inv = pinv(R);
    w = R_inv * Q.transpose() * y;

    z = Q.transpose() * y;

    cout << x << endl;

    cout << w << endl;

    // A and P were used as R and R inverse
    A = R;
    P = R_inv;
    max_obs = max_obs;
    all_Q = Q;
    i = 1;
}

// Update method
void QR_Rls::update(MatrixXld new_x, const MatrixXld &new_y)
{
    // X.conservativeResize(Eigen::NoChange, X.cols() + 1);
    X.conservativeResize(X.rows() + 1, Eigen::NoChange);
    X.row(X.rows() - 1) = new_x;

    y.conservativeResize(new_y.rows() + y.rows(), 1);
    y.bottomRows(new_y.rows()) = new_y;

    int nobs = X.rows();
    P = (1.0 / ff) * P;
    A = ff * A;

    RowVectorXld d = new_x * P;
    MatrixXld c = new_x * (MatrixXld::Identity(A.cols(), A.cols()) - P * A);

    // Update for new regime
    if (!(c.isApprox(RowVectorXld::Zero(c.cols()))))
    {
        MatrixXld c_inv = pinv(c);
        // P = Eigen::MatrixXd::Zero(P.rows(), P.cols() + 1);
        P.leftCols(P.cols()) = P - c_inv * d;
        P.conservativeResize(Eigen::NoChange, P.cols() + 1);
        // MatrixXld m_c_inv_d = c_inv * d;
        // P.leftCols(P.cols() - 1) = P - m_c_inv_d;
        P.col(P.cols() - 1) = c_inv; // Assuming last column is an identity
    }
    // Update for old regime
    else
    {
        VectorXld b_k = (1.0L / (1.0L + d.dot(d))) * P * d.transpose();
        P = MatrixXld::Zero(P.rows(), P.cols() + 1);
        P.leftCols(P.cols() - 1) = P - b_k * d.transpose();
        P.col(P.cols() - 1) = b_k; // Assuming last column is an identity
    }

    A.conservativeResize(A.rows() + 1, A.cols());
    cout << A << endl;
    A.row(A.rows() - 1) = new_x;

    auto [Q, A] = givens_update(this); // Assuming givens is implemented
    // MatrixXld y_reshaped = y;          // Make sure y is in the right shape
    cout << y << endl;
    P = P * Q.transpose();
    w = P * y;

    i++;

    cout << w << endl;

    if (nobs > max_obs)
    {
        // VectorXld x_first = X.col(0);
        downdate();
    }
}

void QR_Rls::downdate()
{
    // Check the condition
    bool temp = (A.transpose() * P.transpose()).isApprox(MatrixXld::Identity(A.cols(), A.cols()));

    // Update matrices
    X = X.rightCols(X.cols() - 1);
    Q = givens_downdate(this); // Assuming a Givens rotation function is defined elsewhere
    P = P * Q;
    A = Q.transpose() * A;

    VectorXld x = A.row(0).transpose().reshaped(dim, 1);
    VectorXld c = MatrixXld::Zero(P.cols(), 1);
    c(0, 0) = 1.0L;
    VectorXld x_T = x.transpose();

    // Deletion for new regime
    if (!temp)
    {
        VectorXld k = P * c;
        VectorXld h = x_T * P;

        P = P - k * pinv(k) * P - P * pinv(h) * h + pinv(k) * P * pinv(h);
    }
    // Deletion for old regime
    else
    {
        VectorXld x_neg_T = -x_T;
        VectorXld h = x_neg_T * P;
        VectorXld u = (MatrixXld::Identity(P.cols(), P.cols()) - A * P) * c;
        VectorXld h_T = h.transpose();
        VectorXld u_T = u.transpose();

        VectorXld k = P * c;
        double h_mag = h.dot(h_T);
        double u_mag = u_T.dot(u);
        double S = 1.0L + (x_neg_T * P * c)(0);
        MatrixXld p_2 = -((u_mag) / S * P * h_T) - k;
        MatrixXld q_2 = -((h_mag) / S * u.transpose() - h);
        double sigma_2 = h_mag * u_mag + S * S;
        P = P + (1 / S) * P * h.transpose() * u.transpose() - (S / sigma_2) * p_2 * q_2;
    }

    P = P.rightCols(P.cols() - 1);
    A = A.bottomRows(A.rows() - 1);

    // Eigen::MatrixXd yMatrix = Eigen::Map<Eigen::MatrixXd>(y.data(), y.size(), 1); // Assuming y is a vector of doubles
    y = all_Q.transpose() * y;
    y = y.bottomRows(y.rows() - 1);

    // Assuming w is a member variable of type Eigen::MatrixXd
    w = P * y;
    all_Q = all_Q.bottomRows(all_Q.rows() - 1);
    // y.erase(y.begin()); // Erase the first element
}

double QR_Rls::pred(const MatrixXld &x)
{
    double pred_value;
    if (x.cols() == 1)
    {
        pred_value = (x.transpose() * w)(0, 0);
    }
    // else
    //{
    //     Eigen::MatrixXd pred_matrix = x.transpose() * w;
    //     pred_value = x.transpose() * self.w;
    // }
    return pred_value;
}

int main()
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXld;

    // Eigen::MatrixXd x(2, 3);
    // x << 1, 2, 3,
    //     4, 5, 6;
    MatrixXld x(2, 10);
    x << 1.0L, 2.0L, 3.0L, 4.0L, 5.0L, 6.0L, 7.0L, 8.0L, 9.0L, 10.0L,
        11.0L, 12.0L, 13.0L, 14.0L, 15.0L, 16.0L, 17.0L, 18.0L, 19.0L, 20.0L;
    VectorXld y(2);
    y << 1.0L, 2.0L;

    MatrixXld up_x(2, 10);
    up_x << 20.0L, 21.5L, 22.0L, 23.0L, 24.0L, 25.0L, 26.0L, 27.0L, 28.0L, 29.0L,
        30.0L, 31.0L, 32.0L, 33.0L, 34.0L, 35.0L, 36.0L, 37.0L, 38.0L, 39.0L;
    VectorXld u_y(2);
    u_y << 3.0L, 4.0L;

    // std::cout << "Type of myMatrix: " << typeid(x).name() << std::endl;

    int max_obs = 10;
    double ff = 1;
    double lambda = 0.1;
    QR_Rls qr_rls(x, y, max_obs, ff, lambda);

    qr_rls.update(up_x.row(0), u_y.row(0));

    return 0;
};