#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "abo/gau_rff.h"

static constexpr double TWO_PI = 6.283185307179586476925286766559;

int GaussianRFF::next_pow2(int x)
{
    if (x <= 1)
        return 1;
    int p = 1;
    while (p < x)
        p <<= 1;
    return p;
}

// Unnormalized Walsh-Hadamard transform (FWHT), O(n log n).
// After two Hadamards, you typically normalize by 1/sqrt(n) overall.
// We’ll fold normalization into the final scale.
void GaussianRFF::hadamard_inplace(Eigen::VectorXd &v)
{
    const int n = (int)v.size();
    for (int len = 1; 2 * len <= n; len <<= 1)
    {
        for (int i = 0; i < n; i += 2 * len)
        {
            for (int j = 0; j < len; ++j)
            {
                const double a = v(i + j);
                const double b = v(i + j + len);
                v(i + j) = a + b;
                v(i + j + len) = a - b;
            }
        }
    }
}

GaussianRFF::GaussianRFF(int d_, int D_, double sigma_, bool seed)
    : d(d_), D(D_), sigma(sigma_)
{
    if (d <= 0 || D <= 0)
        throw std::invalid_argument("d and D must be positive");
    if (sigma <= 0.0)
        throw std::invalid_argument("sigma must be > 0");

    if (seed)
        rng.seed(0);

    d_pad = next_pow2(d);
    n_blocks = (D + d_pad - 1) / d_pad;

    std::normal_distribution<double> gauss01(0.0, 1.0);
    std::uniform_real_distribution<double> unif_phase(0.0, TWO_PI);
    std::bernoulli_distribution radem(0.5);

    blocks.resize(n_blocks);

    for (int blk = 0; blk < n_blocks; ++blk)
    {
        auto &B = blocks[blk];

        // B: +/-1
        B.B.resize(d_pad);
        for (int i = 0; i < d_pad; ++i)
            B.B(i) = radem(rng) ? 1.0 : -1.0;

        // Pi: permutation
        B.Pi.resize(d_pad);
        std::vector<int> idx(d_pad);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int i = 0; i < d_pad; ++i)
            B.Pi(i) = idx[i];

        // G: N(0,1) diagonal
        B.G.resize(d_pad);
        for (int i = 0; i < d_pad; ++i)
            B.G(i) = gauss01(rng);

        // S: scaling diagonal
        // Many implementations use ||g||-style / chi scaling to match radial distribution.
        // A simple, commonly used baseline: S = 1 (or mild normalization).
        // If you want closer-to-paper, you can sample S from chi(d_pad) per coordinate group.
        B.S = Eigen::VectorXd::Ones(d_pad);

        // b: phase per component
        B.b.resize(d_pad);
        for (int i = 0; i < d_pad; ++i)
            B.b(i) = unif_phase(rng);
    }
}

Eigen::MatrixXd GaussianRFF::transform(const Eigen::MatrixXd &x)
{
    if (x.cols() != d)
    {
        throw std::invalid_argument("transform: x.cols() must equal d");
    }

    const int N = (int)x.rows();
    Eigen::MatrixXd Z(N, D);

    // Global scale: account for Hadamard normalization and kernel bandwidth.
    // Since hadamard_inplace is unnormalized, each H contributes a factor sqrt(d_pad).
    // Two H’s => factor d_pad. We compensate by dividing by d_pad, plus sigma*sqrt(d_pad) per Fastfood.
    // Net: (1/(sigma * d_pad * sqrt(d_pad))) * (S H G Π H B x)  (with S=I in this baseline)
    const double fastfood_scale = 1.0 / (sigma * d_pad * std::sqrt((double)d_pad));
    const double out_scale = std::sqrt(2.0 / (double)D);

    Eigen::VectorXd xpad(d_pad), v(d_pad);

    for (int n = 0; n < N; ++n)
    {
        // pad
        xpad.setZero();
        xpad.head(d) = x.row(n).transpose();

        int col_out = 0;
        for (int blk = 0; blk < n_blocks && col_out < D; ++blk)
        {
            const auto &B = blocks[blk];

            v = xpad;

            // v = B ⊙ v
            v.array() *= B.B.array();

            // v = H v
            hadamard_inplace(v);

            // v = Π v
            {
                Eigen::VectorXd tmp = v;
                for (int i = 0; i < d_pad; ++i)
                    v(i) = tmp(B.Pi(i));
            }

            // v = G ⊙ v
            v.array() *= B.G.array();

            // v = H v
            hadamard_inplace(v);

            // v = S ⊙ v (here S=1 baseline)
            v.array() *= B.S.array();

            // scale to approximate omega^T x with omega ~ N(0, 1/sigma^2 I)
            v *= fastfood_scale;

            // Fill output features (cos only, like your original)
            const int take = std::min(d_pad, D - col_out);
            for (int j = 0; j < take; ++j)
            {
                Z(n, col_out + j) = out_scale * std::cos(v(j) + B.b(j));
            }
            col_out += take;
        }
    }

    return Z;
}
