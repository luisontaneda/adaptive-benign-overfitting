#ifndef QRD_RLS_H
#define QRD_RLS_H

#include <cstddef>
#include <Eigen/Dense>
#include "add_row_col.h"

// QRD-RLS with Givens rotations (column-major, C-style arrays)
class QRDRLS
{
public:
    QRDRLS(int max_obs,
           int n_cols,
           double forgetting_factor = 1.0,
           double delta = 1e-2);

    ~QRDRLS();

    void batchInitialize(const double *X_batch,
                         const double *y_batch,
                         int batch_size,
                         int n_cols);

    void update(const double *new_x,
                double new_y,
                double &prediction,
                double &error);

    void downdate();

    void getCoefficients(double *w_out) const;

    size_t getFilterOrder() const { return N_; }
    bool isInitialized() const { return initialized_; }

    double pred(double *x);

    void reset();

private:
    int N_;
    double ff_;
    double sqrt_ff_;
    double delta_;
    int dim_;
    int n_obs_;
    int max_obs_;

    // Pre-allocated fixed-size buffers (no reallocation in hot path)
    double *x;     // dim_
    double *UT_;   // (dim_+1) * (dim_+1), pre-allocated
    double *beta_; // dim_
    double *P_;    // N/A - removed for consistency
    double *R_;    // N/A - removed
    double *Q_;    // N/A - removed
    double *u;     // N/A - removed
    double *X_;    // max_obs_ * dim_, column-major, pre-allocated
    double *y_;    // max_obs_, pre-allocated

    // Ring buffer for efficient windowing (pre-allocated, no reallocation)
    double *X_ring_; // Pre-allocated to max_obs_ * dim_
    double *y_ring_; // Pre-allocated to max_obs_
    int ring_idx_;   // Current position in ring buffer

    bool initialized_;

    inline double &U(int row, int col)
    {
        return UT_[col * N_ + row];
    }

    inline const double &U(int row, int col) const
    {
        return UT_[col * N_ + row];
    }
};

#endif // QRD_RLS_H
