#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "gau_rff.h"
#include <iostream>

using namespace std;

GaussianRFF::GaussianRFF(int d, int D, double kernel_var, bool seed)
    : D(D)
{
    if (seed)
    {
        rng.seed(0); // Fixed seed for reproducibility
    }

    // Initialize random distributions
    normal_distribution<double> normal_dist(0.0, kernel_var);
    uniform_real_distribution<double> uniform_dist(0.0, 2 * M_PI);

    // Initialize A and b matrices
    A = Eigen::MatrixXd(d, D);
    // b = Eigen::VectorXd(D);
    b = Eigen::RowVectorXd(D);

    // Fill A with samples from the normal distribution
    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < D; ++j)
        {
            A(i, j) = normal_dist(rng);
        }
    }

    // Fill b with samples from the uniform distribution
    // for (int j = 0; j < D; ++j)
    for (int j = 0; j < D; ++j)
    {
        b(j) = uniform_dist(rng);
    }
}

Eigen::MatrixXd GaussianRFF::transform_matrix(const Eigen::MatrixXd &x)
{

    // Eigen::MatrixXd b_mat(A.cols(), x.cols());
    Eigen::MatrixXd b_mat(x.rows(), A.cols());

    for (int j = 0; j < x.rows(); ++j)
    {
        b_mat.row(j) = b;
    }

    // std::cout << A.rows() << " " << A.cols() << std::endl;
    // std::cout << x.rows() << " " << x.cols() << std::endl;
    // std::cout << b_mat.rows() << " " << b_mat.cols() << std::endl;

    // Eigen::MatrixXd temp = A.transpose() * x + b_mat;
    Eigen::MatrixXd temp = x * A + b_mat;
    Eigen::MatrixXd z = (std::sqrt(2.0 / D) * temp.array().cos()).matrix();
    return z;
}

Eigen::MatrixXd GaussianRFF::transform(const Eigen::MatrixXd &x)
{
    // Compute z as sqrt(2/D) * cos(A^T * x + b)
    std::cout << A.rows() << " " << A.cols() << std::endl;
    std::cout << x.rows() << " " << x.cols() << std::endl;
    std::cout << b.rows() << " " << b.cols() << std::endl;
    // Eigen::MatrixXd temp = A.transpose() * x + b;
    Eigen::MatrixXd temp = x * A + b;
    Eigen::MatrixXd z = (std::sqrt(2.0 / D) * temp.array().cos()).matrix();
    return z;
}