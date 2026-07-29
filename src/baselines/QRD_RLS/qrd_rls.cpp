#include "baselines/QRD_RLS/qrd_rls.h"
#include "pseudo_inverse.h"
#include "abo/QR_decomposition.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

extern "C"
{
    void drotg_(double *a, double *b, double *c, double *s);
    void dlartg_(double *a, double *b, double *c, double *s, double *r);
    void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
               double *c, double *s);
}

static int drotg_hyp(double a, double b, double *c, double *s, double *r)
{
    if (a == 0.0)
        return 1;
    double aa = fabs(a), bb = fabs(b);
    if (!(aa > bb))
        return 2; // need a^2 > b^2 for real hyperbolic annihilation

    // scale to reduce overflow/underflow risk
    double scale = aa; // since aa > bb, scaling by aa is decent
    double as = a / scale;
    double bs = b / scale;

    double t = as * as - bs * bs; // > 0
    double rs = sqrt(t);          // sqrt((a/scale)^2 - (b/scale)^2)
    double rr = scale * rs;       // sqrt(a^2 - b^2)

    *c = a / rr;
    *s = -b / rr;
    if (r)
        *r = rr;
    return 0;
}

// Apply hyperbolic rotation to vectors x,y (like drot but with hyp params)
static void drot_hyp(int n, double *x, int incx, double *y, int incy,
                     double c, double s)
{
    for (int i = 0; i < n; ++i)
    {
        double xi = x[i * incx];
        double yi = y[i * incy];
        x[i * incx] = c * xi + s * yi;
        y[i * incy] = s * xi + c * yi;
    }
}

void givens_rot(int p, double *v, double *G)
{
    // p = dim_
    // v has length p+1, last index = p
    // G is (p+1)x(p+1) column-major

    int ldG = p + 1;
    int len = p + 1; // number of columns in G
    int inc = ldG;   // step across columns for fixed row in col-major
    int last = p;

    for (int k = 0; k < p; ++k)
    {
        double c, s, r;
        dlartg_(&v[last], &v[k], &c, &s, &r);

        v[last] = r;
        v[k] = 0.0;

        drot_(&len,
              &G[last], &inc, // row "last"
              &G[k], &inc,    // row "k"
              &c, &s);
    }
}

QRDRLS::QRDRLS(int max_obs, int dim_, double forgetting_factor, double delta)
    : max_obs_(max_obs), ff_(forgetting_factor), sqrt_ff_(std::sqrt(forgetting_factor)), delta_(delta), beta_(nullptr), UT_(nullptr), initialized_(false), dim_(dim_), ring_idx_(0)
{
    if (forgetting_factor <= 0.0 || forgetting_factor > 1.0)
    {
        throw std::invalid_argument("Forgetting factor must be in (0, 1]");
    }
    if (delta <= 0.0)
    {
        throw std::invalid_argument("Delta must be positive");
    }

    // Allocate memory - pre-allocated to max_obs_ capacity (no reallocation)
    beta_ = new double[dim_]();

    // Allocate UT_ (dim+1) x (dim+1) for Givens transformations
    UT_ = new double[dim_ * (dim_ + 1)]();

    // Allocate ring buffer and working arrays at full capacity
    X_ring_ = new double[max_obs_ * dim_]();
    y_ring_ = new double[max_obs_]();

    // Allocate an extra row for the append-then-downdate update path
    X_ = new double[(max_obs_ + 1) * dim_]();
    y_ = new double[max_obs_ + 1]();
}

QRDRLS::~QRDRLS()
{
    delete[] beta_;
    delete[] UT_;
    delete[] X_;
    delete[] y_;
    delete[] X_ring_;
    delete[] y_ring_;
}

void QRDRLS::batchInitialize(const double *X_batch, const double *y_batch, int batch_size, int dim)
{
    n_obs_ = batch_size;

    // Copy initial batch to ring buffers
    std::memcpy(X_ring_, X_batch, batch_size * dim_ * sizeof(double));
    std::memcpy(y_ring_, y_batch, batch_size * sizeof(double));
    ring_idx_ = (batch_size % max_obs_);

    // Copy batch data to pre-allocated buffers.
    // X_ uses columns of stride max_obs_+1 so appended row can stay in the buffer.
    for (int j = 0; j < dim_; ++j)
    {
        for (int i = 0; i < batch_size; ++i)
        {
            X_[i + j * (max_obs_ + 1)] = X_batch[i + j * batch_size];
        }
    }
    std::memcpy(y_, y_batch, batch_size * sizeof(double));

    // Compute Q and R from the original contiguous batch data.
    double *Q_temp, *R_temp;
    std::tie(Q_temp, R_temp) = Q_R_compute(const_cast<double *>(X_batch), batch_size, dim_);

    // Compute P = (R^T)^{-1}
    double *P_temp = new double[batch_size * dim_];
    pinv(R_temp, P_temp, batch_size, dim_);

    // Initialize UT_ from P_temp
    for (int j = 0; j < dim_; ++j)
    {
        for (int i = 0; i < dim_; ++i)
        {
            UT_[j + i * (dim_ + 1)] = P_temp[i + j * dim_];
        }
    }

    // Compute Q^T * y_batch for right-hand side
    double z[batch_size];
    cblas_dgemv(CblasColMajor, CblasTrans,
                batch_size, batch_size, 1.0, Q_temp, batch_size, y_batch, 1, 0.0, z, 1);

    cblas_dgemv(CblasColMajor, CblasNoTrans,
                dim_, batch_size, 1.0, P_temp, dim_, z, 1, 0.0, beta_, 1);

    delete[] Q_temp;
    delete[] R_temp;
    delete[] P_temp;
}

void QRDRLS::update(const double *new_x, double new_y, double &prediction, double &error)
{
    double x_oldest[dim_];
    double y_oldest = 0.0;
    bool window_full = (n_obs_ == max_obs_);
    if (window_full)
    {
        std::memcpy(x_oldest, &X_ring_[ring_idx_ * dim_], dim_ * sizeof(double));
        y_oldest = y_ring_[ring_idx_];
    }

    // Store new sample in ring buffers
    std::memcpy(&X_ring_[ring_idx_ * dim_], new_x, dim_ * sizeof(double));
    y_ring_[ring_idx_] = new_y;

    prediction = cblas_ddot(dim_, new_x, 1, beta_, 1);
    error = new_y - prediction;

    int dim_p_1 = dim_ + 1;

    double a[dim_p_1];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                dim_p_1, dim_, -1.0, UT_, dim_p_1, new_x, 1, 0.0, a, 1);

    a[dim_] = 1;
    double G[dim_p_1 * dim_p_1] = {0};
    for (int i = 0; i < dim_p_1; i++)
    {
        G[i * dim_p_1 + i] = 1;
    }

    givens_rot(dim_, a, G);

    double result_1[dim_ * dim_p_1];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                dim_p_1, dim_, dim_p_1, 1, G, dim_p_1, UT_, dim_p_1, 0, result_1, dim_p_1);
    std::memcpy(UT_, result_1, dim_ * dim_p_1 * sizeof(double));

    double delta = a[dim_];

    for (int i = 0; i < dim_; i++)
    {
        double u = UT_[i * dim_p_1 + dim_];
        beta_[i] -= (new_y - prediction) / delta * u;
        UT_[i * dim_p_1 + dim_] = 0;
    }

    // Append new X sample and y to pre-allocated buffer in column-major layout.
    // Use max_obs_+1 as the stride because X_ stores one extra row for the append/downdate path.
    for (int j = 0; j < dim_; ++j)
    {
        X_[n_obs_ + j * (max_obs_ + 1)] = new_x[j];
    }
    y_[n_obs_] = new_y;

    n_obs_++;

    // Advance ring buffer position
    ring_idx_ = (ring_idx_ + 1) % max_obs_;

    if (n_obs_ > max_obs_)
    {
        downdate(x_oldest, y_oldest);
    }
}

void QRDRLS::downdate(const double *x_oldest, double y_oldest)
{
    // The oldest sample is supplied by the caller and preserved before the ring buffer overwrite.
    double x_T[dim_];
    for (int i = 0; i < dim_; ++i)
    {
        x_T[i] = X_[i * (max_obs_ + 1)]; // Copy the first row (column-major stride)
    }

    int dim_p_1 = dim_ + 1;

    double b[dim_p_1];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                dim_p_1, dim_, -1.0, UT_, dim_p_1, x_T, 1, 0.0, b, 1);

    b[dim_] = 1;

    double G[dim_p_1 * dim_p_1] = {0};
    for (int i = 0; i < dim_p_1; i++)
    {
        G[i * dim_p_1 + i] = 1;
    }

    double c, s, r;

    int p = dim_;
    int ldG = p + 1;
    int len = p + 1;
    int inc = ldG;
    int last = p;

    for (int k = 0; k < p; ++k)
    {
        // Build hyperbolic rotation to zero b[k] using b[last]
        double c, s, r;
        int rc = drotg_hyp(b[last], b[k], &c, &s, &r);

        b[last] = r;
        b[k] = 0.0;

        // Accumulate into G by rotating rows (last, k) across all columns
        drot_hyp(len, &G[last], inc, &G[k], inc, c, s);
    }

    double result_1[dim_ * dim_p_1];

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                dim_p_1, dim_, dim_p_1, 1, G, dim_p_1, UT_, dim_p_1, 0, result_1, dim_p_1);
    std::memcpy(UT_, result_1, dim_ * dim_p_1 * sizeof(double));

    double temp_pred = pred(x_T);
    double delta = b[dim_];

    for (int i = 0; i < dim_; i++)
    {
        double u = UT_[i * dim_p_1 + dim_];
        beta_[i] -= (y_[0] - temp_pred) / delta * u;
        UT_[i * dim_p_1 + dim_] = 0;
    }

    // Shift X: move rows 1..n_obs_-1 to rows 0..n_obs_-2 (column-major)
    for (int j = 0; j < dim_; ++j)
    {
        for (int i = 0; i < n_obs_ - 1; ++i)
        {
            X_[i + j * (max_obs_ + 1)] = X_[i + 1 + j * (max_obs_ + 1)];
        }
    }

    // Shift y: move y[1]..y[n_obs_-1] to y[0]..y[n_obs_-2]
    for (int i = 0; i < n_obs_ - 1; ++i)
    {
        y_[i] = y_[i + 1];
    }

    n_obs_--;
}

double QRDRLS::pred(double *x)
{
    double pred_value = cblas_ddot(dim_, x, 1, beta_, 1);
    return pred_value;
}

void QRDRLS::reset()
{
    std::memset(beta_, 0, (n_obs_ + 1) * sizeof(double));
    std::memset(UT_, 0, (n_obs_ + 1) * (n_obs_ + 1) * sizeof(double));
    initialized_ = false;
}

void QRDRLS::getCoefficients(double *w_out) const
{
    std::memcpy(w_out, beta_, dim_ * sizeof(double));
}
