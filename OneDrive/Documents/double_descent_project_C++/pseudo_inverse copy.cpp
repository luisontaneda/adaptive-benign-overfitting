#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>

typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> VectorXld;

MatrixXld pseudoInverse(const MatrixXld &a, long double tolerance = 1e-16)
{

    // Create a Householder QR preconditioner
    Eigen::HouseholderQR<MatrixXld> preconditioner(a);

    // Apply the preconditioner to the matrix
    MatrixXld preconditioned_A = preconditioner.matrixQR();

    Eigen::BDCSVD<MatrixXld> svd(preconditioned_A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Eigen::BDCSVD<MatrixXld> svd(a, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Perform SVD
    MatrixXld U = svd.matrixU();
    MatrixXld V = svd.matrixV();
    VectorXld S_v_inv = svd.singularValues();

    std::cout << U << std::endl;

    // decltype(U(0, 0)) elementType;
    //  Output the type as a string (this works at runtime if you want to print the type)
    // std::cout << "Type of matrix element: " << typeid(elementType).name() << std::endl;

    for (int i = 0; i < S_v_inv.size(); ++i)
    {
        if (S_v_inv(i) > tolerance)
        {
            S_v_inv(i) = 1.0L / S_v_inv(i);
        }
        else
        {
            S_v_inv(i) = 0.0L; // Singular values below the threshold are treated as zero
        }
    }

    MatrixXld S_inv = MatrixXld::Zero(U.cols(), V.rows());
    S_inv.diagonal().head(S_v_inv.size()) = S_v_inv;
    MatrixXld U_T = U.transpose();

    std::cout << V * S_inv.transpose() * U_T << std::endl;

    return V * S_inv.transpose() * U_T;
}