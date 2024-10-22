#include <iostream>
#include <Eigen/Dense>
#include <cmath>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXld;

std::pair<MatrixXld, MatrixXld> Givens_Rotation(const MatrixXld &R)
{
    int n = R.rows();
    MatrixXld Q = MatrixXld::Identity(n, n);
    MatrixXld R_copy = R;

    for (int j = 0; j < R.cols(); ++j)
    {
        for (int i = j + 1; i < n; ++i)
        {
            double x = R_copy(j, j);
            double y = R_copy(i, j);
            double r = std::sqrt(x * x + y * y);

            if (r != 0)
            {
                double c = x / r;
                double s = -y / r;

                // Create a Givens rotation matrix
                MatrixXld I = MatrixXld::Identity(n, n);
                I(i, i) = c;
                I(j, j) = c;
                I(i, j) = s;
                I(j, i) = -s;

                // Apply rotation
                Q = Q * I.transpose();
                R_copy = I * R_copy;
            }
        }
    }

    return {Q, R_copy};
}