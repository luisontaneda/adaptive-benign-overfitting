#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "QR_decomposition.h"

using namespace std;

class GaussianRFF
{
public:
   Eigen::MatrixXd A;
   Eigen::RowVectorXd b;
   Eigen::MatrixXd A_ort;
   int D;
   mutable std::mt19937 rng; // Random number generator

   GaussianRFF(int d, int D, double kernel_var, bool seed);
   ~GaussianRFF();
   Eigen::MatrixXd transform_matrix(const Eigen::MatrixXd &x);
   Eigen::MatrixXd transform(const Eigen::MatrixXd &x);
};
