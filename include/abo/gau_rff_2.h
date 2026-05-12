#pragma once
#include <Eigen/Dense>
#include <random>
#include <vector>

class GaussianRFF
{
public:
   GaussianRFF(int d, int D, double sigma, bool seed);

   Eigen::MatrixXd transform(const Eigen::MatrixXd &x);
   Eigen::MatrixXd transform_matrix(const Eigen::MatrixXd &x) { return transform(x); }

private:
   int d;        // original dim
   int D;        // #features
   int d_pad;    // next power of two >= d
   int n_blocks; // ceil(D / d_pad)
   double sigma;

   std::mt19937 rng;

   struct Block
   {
      Eigen::VectorXd B;  // +/-1 (diag)
      Eigen::VectorXi Pi; // permutation indices
      Eigen::VectorXd G;  // N(0,1) (diag)
      Eigen::VectorXd S;  // scaling (diag)
      Eigen::VectorXd b;  // phase in [0,2pi)
   };

   std::vector<Block> blocks;

   static int next_pow2(int x);
   static void hadamard_inplace(Eigen::VectorXd &v); // FWHT (unnormalized)
};
