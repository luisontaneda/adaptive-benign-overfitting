#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "abo/gau_rff.h"
#include <iostream>

using namespace std;

// NOTE:
// - This version implements BLOCK ORTHOGONAL RANDOM FEATURES (ORF) for D > d.
// - It does NOT use Q_R_compute / Q_ / R_. You can remove those from your header.
// - It generates W = A_ort (d x D) in blocks of size d, using QR on Gaussian blocks,
//   then chi-scaling per column to match Gaussian norms.
// - kernel_var handling: your old code used normal_dist(0, kernel_var) where kernel_var
//   was treated as stddev (C++ uses stddev). Here we generate N(0,1) then scale W by kernel_var.
//   If your kernel_var is variance, change scale = sqrt(kernel_var).

GaussianRFF::~GaussianRFF() = default;

GaussianRFF::GaussianRFF(int d, int D, double kernel_var, bool seed)
    : D(D)
{
    if (seed)
    {
        rng.seed(0);
    }

    // Distributions
    std::normal_distribution<double> normal01(0.0, 1.0);
    std::chi_squared_distribution<double> chi2(d);
    std::uniform_real_distribution<double> uniform_dist(0.0, 2.0 * M_PI);

    // Allocate
    A_ort = Eigen::MatrixXd(d, D);
    b = Eigen::RowVectorXd(D);

    // ---- Build block-ORF matrix A_ort (d x D) ----
    // Each block:
    //   G ~ N(0,1)^{dxd}
    //   QR: G = Q R
    //   Fix signs to get Haar-like Q (recommended)
    //   radii r_i = sqrt(chi2(d))  (i.e., chi(d))
    //   W_block = Q * diag(r)
    //
    // Then concatenate blocks until we fill D columns.

    int col = 0;
    while (col < D)
    {
        // 1) Gaussian block G
        Eigen::MatrixXd G(d, d);
        for (int i = 0; i < d; ++i)
        {
            for (int j = 0; j < d; ++j)
            {
                G(i, j) = normal01(rng);
            }
        }

        // 2) QR => Q
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(G);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(d, d);

        // 2b) Sign correction using diag(R) to better match Haar orthogonal distribution
        Eigen::MatrixXd R = qr.matrixQR().template triangularView<Eigen::Upper>();
        for (int i = 0; i < d; ++i)
        {
            double s = (R(i, i) >= 0.0) ? 1.0 : -1.0;
            Q.col(i) *= s;
        }

        // 3) Radii (chi(d)) per column
        Eigen::VectorXd r(d);
        for (int i = 0; i < d; ++i)
        {
            r(i) = std::sqrt(chi2(rng));
        }

        // 4) Fill into A_ort (truncate last block if needed)
        int take = std::min(d, D - col);
        for (int j = 0; j < take; ++j)
        {
            A_ort.col(col + j) = Q.col(j) * r(j);
        }

        col += take;
    }

    // ---- Kernel scaling ----
    // Your original code used normal_distribution(0, kernel_var) which interprets kernel_var as STDDEV.
    // So to match that behavior, scale the ORF matrix by kernel_var.
    //
    // If instead kernel_var is actually a variance in your codebase, use:
    //   double scale = std::sqrt(kernel_var);
    // below.
    double scale = kernel_var;
    A_ort *= scale;

    // ---- Random phase ----
    for (int j = 0; j < D; ++j)
    {
        b(j) = uniform_dist(rng);
    }

    // (Optional) keep A as alias for compatibility if other code expects it
    A = A_ort;
}

Eigen::MatrixXd GaussianRFF::transform_matrix(const Eigen::MatrixXd &x)
{
    // z = sqrt(2/D) * cos(x * A_ort + b)
    // x: (n x d), A_ort: (d x D), b: (1 x D)
    Eigen::MatrixXd temp = (x * A_ort).rowwise() + b;
    Eigen::MatrixXd z = (std::sqrt(2.0 / D) * temp.array().cos()).matrix();
    return z;
}

Eigen::MatrixXd GaussianRFF::transform(const Eigen::MatrixXd &x)
{
    // Same as transform_matrix; keep both if your interface expects them.
    Eigen::MatrixXd temp = (x * A_ort).rowwise() + b;
    Eigen::MatrixXd z = (std::sqrt(2.0 / D) * temp.array().cos()).matrix();
    return z;
}
