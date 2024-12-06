#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <string>

#ifndef QR_RLS_H // Check if QR_RLS_H is not defined
#define QR_RLS_H // Define QR_RLS_H

using namespace std;

class QR_Rls
{
public:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;
    typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVectorXd;

    MatrixXd X, I, B, Q, R, R_inv, A, P, all_Q;
    VectorXd y, w, z;
    int max_obs, dim, n_batch, i;
    double ff, l, b;

    QR_Rls(const MatrixXd &x, const VectorXd &y, int max_obs, double ff, double lambda);

    // Update method
    void update(MatrixXd new_x, const MatrixXd &y);

    void downdate();

    double pred(const MatrixXd &x);
};

#endif // QR_RLS_H  // End of the include guard