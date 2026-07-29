#include "baselines/KRLS_RBF/krls_rbf.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <cstddef>
#include <cmath>

bool isSymmetricColMajor(const double *A, int n, double tol = 1e-12)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = j + 1; i < n; ++i)
        {
            double diff = std::abs(A[i + j * n] - A[j + i * n]);
            if (diff > tol)
                return false;
        }
    }
    return true;
}

// Constructor
KRLS_RBF::KRLS_RBF(const double *X_init, const double *y_init, int n_obs,
                   int dim_, double delta, double sigma,
                   int window_size)
    : dim_(dim_),
      n_obs_(n_obs),
      window_size_(window_size),
      delta_(delta),
      sigma_(sigma),
      initialized_(false),
      X_init_(nullptr),
      y_init_(nullptr),
      n_init_samples_(0),
      ring_idx_(0)
{

    // Pre-allocate all buffers to window_size_ capacity (no reallocation in hot path)
    beta_ = new double[window_size_]();
    P_ = new double[window_size_ * window_size_]();

    // Allocate ring buffers (pre-allocated to window_size_)
    X_ring_ = new double[window_size_ * dim_]();
    y_ring_ = new double[window_size_]();

    // Allocate working arrays at full capacity
    h_ = new double[window_size_]();
    X_ = new double[window_size_ * dim_]();
    y_ = new double[window_size_]();
    K_ = new double[window_size_ * window_size_]();

    // Store initial data if provided
    if (X_init != nullptr && y_init != nullptr && n_obs_ > 0)
    {
        n_init_samples_ = n_obs_;
        X_init_ = new double[window_size_ * dim_]();
        y_init_ = new double[n_obs_];

        for (int j = 0; j < dim_; ++j)
        {
            for (int i = 0; i < n_obs_; ++i)
            {
                X_init_[i + j * window_size_] = X_init[i + j * n_obs_];
                X_[i + j * window_size_] = X_init[i + j * n_obs_];
            }
        }
        vectorCopy(y_init_, y_init, n_obs_);
        vectorCopy(y_, y_init, n_obs_);

        // Copy initial batch to ring buffers in row-major order
        for (int i = 0; i < n_obs_; ++i)
        {
            for (int j = 0; j < dim_; ++j)
            {
                X_ring_[i * dim_ + j] = X_init[i + j * n_obs_];
            }
        }
        std::memcpy(y_ring_, y_init, n_obs_ * sizeof(double));
        ring_idx_ = (n_obs_ % window_size_);

        // Initialize from batch data
        initializeFromBatch(X_, y_, n_obs_);
    }
}

// Destructor
KRLS_RBF::~KRLS_RBF()
{
    delete[] X_;
    delete[] y_;
    delete[] beta_;
    delete[] P_;
    delete[] h_;
    delete[] K_;
    delete[] X_ring_;
    delete[] y_ring_;

    if (X_init_ != nullptr)
    {
        delete[] X_init_;
    }
    if (y_init_ != nullptr)
    {
        delete[] y_init_;
    }
}

// RBF kernel for vectors with a stride (stride=1 means contiguous)
double KRLS_RBF::kernel(const double *x1, int stride1,
                        const double *x2, int stride2) const
{
    double sum_sq = 0.0;
    for (int k = 0; k < dim_; ++k)
    {
        double diff = x1[k * stride1] - x2[k * stride2];
        sum_sq += diff * diff;
    }
    return std::exp(-sum_sq / (2.0 * sigma_ * sigma_));
}

// Copy vector
void KRLS_RBF::vectorCopy(double *dest, const double *src, int n)
{
    std::memcpy(dest, src, n * sizeof(double));
}

// Initialize from batch of data
void KRLS_RBF::initializeFromBatch(const double *X, const double *y, int n_obs_)
{

    // Build kernel matrix K

    for (int j = 0; j < n_obs_; ++j)
    {
        const double *xj0 = &X[j];
        for (int i = 0; i < n_obs_; ++i)
        {
            const double *xi0 = &X[i];

            K_[i + j * window_size_] = kernel(xi0, window_size_, xj0, window_size_);

            if (i == j)
                K_[i + j * window_size_] += delta_;
        }
    }

    // Compute Q = (K + λI)^(-1)
    pinv(K_, P_, n_obs_, n_obs_);

    cblas_dgemv(CblasColMajor, CblasNoTrans,
                n_obs_, n_obs_, 1.0, P_, window_size_, y_, 1, 0.0, beta_, 1);

    initialized_ = true;
}

// Update with new sample
void KRLS_RBF::update(const double *new_x, double new_y, double &prediction, double &error)
{
    // Remove the oldest sample when the buffer is full before adding a new one
    if (n_obs_ == window_size_)
    {
        downdate();
    }

    // Store new sample in ring buffers
    std::memcpy(&X_ring_[ring_idx_ * dim_], new_x, dim_ * sizeof(double));
    y_ring_[ring_idx_] = new_y;

    // Compute h(i): vector of kernel evaluations with existing samples
    for (int j = 0; j < n_obs_; ++j)
    {
        const double *xj0 = &X_[j];
        h_[j] = kernel(new_x, 1, xj0, window_size_);

        // Update K with new row/column
        K_[j + n_obs_ * window_size_] = h_[j]; // New row, old columns
        K_[n_obs_ + j * window_size_] = h_[j]; // New column, old rows
    }

    double d_k = kernel(new_x, 1, new_x, 1) + delta_;
    K_[n_obs_ + n_obs_ * window_size_] = d_k; // Diagonal element

    prediction = cblas_ddot(n_obs_, h_, 1, beta_, 1);
    error = new_y - prediction;

    // Store new X sample in column-major layout
    for (int k = 0; k < dim_; ++k)
    {
        X_[n_obs_ + k * window_size_] = new_x[k];
    }

    // Update of inverse matrix
    double P_b[window_size_];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                n_obs_, n_obs_, 1.0, P_, window_size_, h_, 1, 0.0, P_b, 1);
    double g = 1.0 / (d_k - cblas_ddot(n_obs_, h_, 1, P_b, 1));

    cblas_dger(CblasColMajor, n_obs_, n_obs_, g, P_b, 1, P_b, 1, P_, window_size_);

    // Append new row/column to P (write at position n_obs_)
    for (int j = 0; j < n_obs_; ++j)
    {
        P_[j + n_obs_ * window_size_] = -g * P_b[j];
        P_[n_obs_ + j * window_size_] = -g * P_b[j];
    }
    P_[n_obs_ + n_obs_ * window_size_] = g;

    y_[n_obs_] = new_y;

    n_obs_++;

    // Update coefficients for the new augmented system
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                n_obs_, n_obs_, 1.0, P_, window_size_, y_, 1, 0.0, beta_, 1);

    // Advance ring buffer position
    ring_idx_ = (ring_idx_ + 1) % window_size_;
}

double KRLS_RBF::predict(const double *x) const
{
    if (n_obs_ <= 0)
        return 0.0;

    double prediction = 0.0;
    for (int j = 0; j < n_obs_; ++j)
    {
        const double *xj0 = &X_[j];
        double k = kernel(x, 1, xj0, window_size_);
        prediction += beta_[j] * k;
    }
    return prediction;
}

void KRLS_RBF::downdate()
{
    // Shift all K and P rows/columns down by one (remove oldest sample)
    // K and P are window_size_ x window_size_ column-major

    // Shift K: move rows 1..n_obs_-1 to rows 0..n_obs_-2
    for (int j = 0; j < n_obs_; ++j)
    {
        for (int i = 0; i < n_obs_ - 1; ++i)
        {
            K_[i + j * window_size_] = K_[i + 1 + j * window_size_];
        }
    }
    // Shift K columns: move columns 1..n_obs_-1 to columns 0..n_obs_-2
    for (int j = 1; j < n_obs_; ++j)
    {
        for (int i = 0; i < n_obs_ - 1; ++i)
        {
            K_[i + (j - 1) * window_size_] = K_[i + j * window_size_];
        }
    }

    // Extract f vector from P before shift
    double f[window_size_];
    for (int i = 0; i < n_obs_ - 1; i++)
    {
        f[i] = P_[i + 1];
    }
    double e = P_[0];

    // Shift P: move rows 1..n_obs_-1 to rows 0..n_obs_-2
    for (int j = 0; j < n_obs_; ++j)
    {
        for (int i = 0; i < n_obs_ - 1; ++i)
        {
            P_[i + j * window_size_] = P_[i + 1 + j * window_size_];
        }
    }
    // Shift P columns: move columns 1..n_obs_-1 to columns 0..n_obs_-2
    for (int j = 1; j < n_obs_; ++j)
    {
        for (int i = 0; i < n_obs_ - 1; ++i)
        {
            P_[i + (j - 1) * window_size_] = P_[i + j * window_size_];
        }
    }

    // Shift y: move y[1]..y[n_obs_-1] to y[0]..y[n_obs_-2]
    for (int i = 0; i < n_obs_ - 1; ++i)
    {
        y_[i] = y_[i + 1];
    }

    // Shift X: move rows 1..n_obs_-1 to rows 0..n_obs_-2
    for (int j = 0; j < dim_; ++j)
    {
        for (int i = 0; i < n_obs_ - 1; ++i)
        {
            X_[i + j * window_size_] = X_[i + 1 + j * window_size_];
        }
    }

    n_obs_--;

    // Rank-1 update to P
    cblas_dger(CblasColMajor, n_obs_, n_obs_, -1.0 / e, f, 1, f, 1, P_, window_size_);

    // Update beta from P and y
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                n_obs_, n_obs_, 1.0, P_, window_size_, y_, 1, 0.0, beta_, 1);
}

// Reset filter
void KRLS_RBF::reset()
{
    if (X_init_ != nullptr && y_init_ != nullptr)
    {
        initializeFromBatch(X_init_, y_init_, n_init_samples_);
    }
    else
    {
        initialized_ = false;
        std::memset(beta_, 0, window_size_ * sizeof(double));
        std::memset(P_, 0, window_size_ * window_size_ * sizeof(double));
    }
}
