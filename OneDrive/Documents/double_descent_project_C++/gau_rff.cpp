#include <Eigen/Dense>
#include <cmath>
#include <random>

using namespace std;

class GaussianRFF
{
public:
    GaussianRFF(int d, int D, double kernel_var = 1.0, bool seed = true)
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
        b = Eigen::VectorXd(D);

        // Fill A with samples from the normal distribution
        for (int i = 0; i < d; ++i)
        {
            for (int j = 0; j < D; ++j)
            {
                A(i, j) = normal_dist(rng);
            }
        }

        // Fill b with samples from the uniform distribution
        for (int j = 0; j < D; ++j)
        {
            b(j) = uniform_dist(rng);
        }
    }

    Eigen::VectorXd transform(const Eigen::VectorXd &x) const
    {
        // Compute z as sqrt(2/D) * cos(A^T * x + b)
        Eigen::VectorXd temp = A.transpose() * x + b;
        Eigen::VectorXd z = (std::sqrt(2.0 / D) * temp.array().cos()).matrix();
        return z;
    }

private:
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    int D;
    mutable std::mt19937 rng; // Random number generator
};