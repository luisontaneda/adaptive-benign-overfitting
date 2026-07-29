#include <algorithm> // std::fill
#include <cmath>
#include <cstring>
#include "abo/last_row_givens.h"

extern "C"
{
   void dlartg_(double *a, double *b, double *c, double *s, double *r);
   void drot_(int *n, double *dx, int *incx, double *dy, int *incy,
              double *c, double *s);
}

namespace givens
{

   void update(ABO *abo)
   {
      double c, s, r;
      int n_obs   = abo->n_obs_;   // current size AFTER n_obs_++ in caller
      int dim     = abo->dim_;
      int max_obs = abo->max_obs_;

      double *R     = abo->R_;
      double *R_inv = abo->R_inv_;

      int col_stride = max_obs;        // R_ column stride
      int last = n_obs - 1;            // index of the newly added row

      int limit = std::min(n_obs - 1, dim);

      for (int j = 0; j < limit; ++j)
      {
         // Zero R(last, j) using Givens on rows j and last
         dlartg_(&R[j + j * col_stride],
                 &R[last + j * col_stride],
                 &c, &s, &r);

         R[j + j * col_stride]    = r;
         R[last + j * col_stride] = 0.0;

         // Apply rotation to remaining columns j+1..dim-1 of rows j and last
         int temp  = dim - j - 1;
         int idx_1 = j    + (j + 1) * col_stride;
         int idx_2 = last + (j + 1) * col_stride;

         drot_(&temp, &R[idx_1], &col_stride,
               &R[idx_2], &col_stride,
               &c, &s);

         // Apply same rotation to R_inv_ columns j and last
         int inc = 1;
         drot_(&dim, &R_inv[j * dim], &inc, &R_inv[last * dim], &inc, &c, &s);
      }
   }

   // Downdate: uses h = R_inv^T * z_old (Q-less, correct for both regimes).
   void downdate(ABO *abo, double *z_old)
   {
      int n_obs   = abo->n_obs_;   // current size BEFORE n_obs_-- in caller
      int dim     = abo->dim_;

      double *R     = abo->R_;
      double *G     = abo->G_;
      double *R_inv = abo->R_inv_;
      double *h     = abo->scratch_n_;  // n_obs-length working vector

      cblas_dgemv(CblasColMajor, CblasTrans,
                     dim, n_obs, 1.0,
                     R_inv, dim, z_old, 1, 0.0, h, 1);

      // Place the residual in the first 'empty' slot of the h vector
      // This completes the first row of Q (q1)
      if (dim < n_obs) {
         // 2. Find the norm of the 'known' part
         double norm_sq = cblas_ddot(dim, h, 1, h, 1);
         double rho = std::sqrt(std::max(0.0, 1.0 - norm_sq));
         h[dim] = rho;
         // Ensure the rest of h (up to n_obs) is zero
         std::fill(h + dim + 1, h + n_obs, 0.0);
      }

      std::fill(G, G + n_obs * n_obs, 0.0);
          for (int i = 0; i < n_obs; i++) G[i * n_obs + i] = 1.0;

      double c, s, r;
      int one = 1;

      // Step 3: Givens rotations to zero h[1..n_obs-1] bottom-up
      for (int i = n_obs - 1; i > 0; --i)
      {
         dlartg_(&h[i - 1], &h[i], &c, &s, &r);
         h[i - 1] = r;
         h[i]     = 0.0;

         // Rotate elements i-1 and i of our accumulated first row of Q
         drot_(&n_obs, &G[(i - 1) * n_obs], &one, &G[i * n_obs], &one, &c, &s);

         if (dim > n_obs)
         {
            // New regime: rotate rows i-1 and i of R (stride max_obs)
            int n   = dim - i + 1;
            int inc = abo->max_obs_;
            drot_(&n,
                  &R[(i - 1) * abo->max_obs_ + (i - 1)], &inc,
                  &R[(i - 1) * abo->max_obs_ +  i      ], &inc,
                  &c, &s);

            abo->giv_rots.push_back({(i - 1) * dim, i * dim, c, s});
         }
      }

      // Step 4: sign fix and finalize
      if (dim <= n_obs)
      {
         // Old regime: sign fix using h[0] after rotations
         if (h[0] < 0)
         {
            for (int i = 0; i < n_obs * n_obs; ++i) G[i] *= -1;
         }
      }

      for (int t = 0; t < n_obs; t++)
         {
            abo->G_e_1_[t] = G[t];
         }

      if (dim <= n_obs)
      {
         // We apply G^T * R directly.
         // G is (n_obs x n_obs)
         // R is (n_obs x dim) but embedded in a (max_obs x dim) buffer.

         // Use scratch_d2_ as a temporary destination for the result 
         // to avoid overwriting R while it's being read.
         cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                     n_obs,          // M: rows of op(G) and result
                     dim,            // N: columns of R and result
                     n_obs,          // K: columns of op(G) and rows of R
                     1.0, 
                     G, n_obs,       // LDA of G is n_obs
                     R, abo->max_obs_, // LDA of R is max_obs_ (CRITICAL)
                     0.0, 
                     abo->scratch_d2_, n_obs); // LDA of result is n_obs

         // Now copy the result back into the R buffer, respecting the stride
         for (int j = 0; j < dim; j++)
         {
            std::memcpy(&R[j * abo->max_obs_], 
                        &abo->scratch_d2_[j * n_obs], 
                        n_obs * sizeof(double));
            
            // Zero out the deleted row (the last row of the new n_obs-1 system)
            // or the 'residual' row created by the downdate if necessary.
            R[j * abo->max_obs_ + (n_obs - 1)] = 0.0; 
         }
      }
   }

} // namespace givens
