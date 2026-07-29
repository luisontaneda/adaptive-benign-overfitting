#include "abo/dd_test.h"
#include "baselines/QRD_RLS/qrd_rls.h"
#include "baselines/KRLS_RBF/krls_rbf.h"
#include "abo/QR_decomposition.h"

#include <Eigen/Dense>
#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

// ---- data structures ----

struct ModelParamsABO
{
   int L = -1, W = -1;
   double sigma = -1.0;
   int D = 2048;
   int log2D = -1;
   double ff = 1.0;
   double regularizer = 1e-2;
};

struct ModelParamsQRD
{
   int L = -1, W = -1;
   double ff = 1.0;
   double regularizer = 1e-2;
};

struct ModelParamsKRLS
{
   int L = -1, W = -1;
   double sigma = -1.0;
   double ff = 1.0;
   double regularizer = 1e-2;
};

struct CommonParams
{
   int first_date = 960 * 8;
   int start_k = 0;
   int end_k = 5; // exclusive
   int val_length = 960 * 2;
   std::string out_csv = "results/gridsearch/EURUSD/best_test.csv";

   bool run_abo = true;
   bool run_qrd = true;
   bool run_krls = true;
};

struct Args
{
   CommonParams common;
   ModelParamsABO abo;
   ModelParamsQRD qrd;
   ModelParamsKRLS krls;
};

struct FoldResultRow
{
   int fold = 0;
   std::string model;

   int L = 0, W = 0;
   double sigma = std::numeric_limits<double>::quiet_NaN();
   int D = 0;

   double mse = std::numeric_limits<double>::quiet_NaN();
   double var = std::numeric_limits<double>::quiet_NaN();
   bool valid = false;
};

// ---- helpers ----

static inline bool is_flag(const char *a, const char *b)
{
   return std::strcmp(a, b) == 0;
}

static inline void parse_args(int argc, char **argv, Args &a)
{
   for (int i = 1; i < argc; ++i)
   {
      auto need = [&](const char *flag)
      {
         if (i + 1 >= argc)
            throw std::runtime_error(std::string("Missing value for ") + flag);
      };

      // ---- common ----
      if (is_flag(argv[i], "--first_date"))
      {
         need("--first_date");
         a.common.first_date = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--start_k"))
      {
         need("--start_k");
         a.common.start_k = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--end_k"))
      {
         need("--end_k");
         a.common.end_k = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--val_length"))
      {
         need("--val_length");
         a.common.val_length = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--out_csv"))
      {
         need("--out_csv");
         a.common.out_csv = argv[++i];
      }
      else if (is_flag(argv[i], "--run"))
      {
         need("--run");
         std::string s = argv[++i];
         auto has = [&](const std::string &key) { return s.find(key) != std::string::npos; };
         a.common.run_abo = has("abo");
         a.common.run_qrd = has("qrd");
         a.common.run_krls = has("krls");
      }

      // ---- ABO ----
      else if (is_flag(argv[i], "--abo_lags"))
      {
         need("--abo_lags");
         a.abo.L = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--abo_window"))
      {
         need("--abo_window");
         a.abo.W = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--abo_sigma"))
      {
         need("--abo_sigma");
         a.abo.sigma = std::stod(argv[++i]);
      }
      else if (is_flag(argv[i], "--abo_D"))
      {
         need("--abo_D");
         a.abo.D = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--abo_log2D"))
      {
         need("--abo_log2D");
         a.abo.log2D = std::stoi(argv[++i]);
      }

      // ---- QRD ----
      else if (is_flag(argv[i], "--qrd_lags"))
      {
         need("--qrd_lags");
         a.qrd.L = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--qrd_window"))
      {
         need("--qrd_window");
         a.qrd.W = std::stoi(argv[++i]);
      }

      // ---- KRLS ----
      else if (is_flag(argv[i], "--krls_lags"))
      {
         need("--krls_lags");
         a.krls.L = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--krls_window"))
      {
         need("--krls_window");
         a.krls.W = std::stoi(argv[++i]);
      }
      else if (is_flag(argv[i], "--krls_sigma"))
      {
         need("--krls_sigma");
         a.krls.sigma = std::stod(argv[++i]);
      }
   }

   if (a.abo.log2D >= 0)
      a.abo.D = 1 << a.abo.log2D;

   if (a.common.end_k <= a.common.start_k)
      throw std::runtime_error("--end_k must be > --start_k");

   if (!a.common.run_abo && !a.common.run_qrd && !a.common.run_krls)
      throw std::runtime_error("--run must include at least one of: abo,qrd,krls");

   if (a.common.run_abo)
   {
      if (a.abo.L <= 0 || a.abo.W <= 0 || a.abo.sigma <= 0.0 || a.abo.D <= 0)
         throw std::runtime_error("ABO needs --abo_lags --abo_window --abo_sigma and --abo_D/--abo_log2D");
   }
   if (a.common.run_qrd)
   {
      if (a.qrd.L <= 0 || a.qrd.W <= 0)
         throw std::runtime_error("QRD needs --qrd_lags --qrd_window");
   }
   if (a.common.run_krls)
   {
      if (a.krls.L <= 0 || a.krls.W <= 0 || a.krls.sigma <= 0.0)
         throw std::runtime_error("KRLS needs --krls_lags --krls_window --krls_sigma");
   }
}

static inline void lag_matrix(const std::vector<double> &x, int lag, std::vector<std::vector<double>> &X_lag, std::vector<double> &y)
{
   const int T = static_cast<int>(x.size());
   const int N = T - lag;
   if (N <= 0)
      return;
   X_lag.assign(N, std::vector<double>(lag));
   y.assign(N, 0.0);
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < lag; ++j)
         X_lag[i][j] = x[i + j];
      y[i] = x[i + lag];
   }
}

static inline void dataset_creation(std::vector<std::vector<double>> &data_set, std::vector<double> &target_data, Eigen::MatrixXd &initial_matrix, Eigen::MatrixXd &update_matrix, double *y, double *&y_update, int num_rows, int num_cols, int start_row)
{
   for (int i = 0; i < num_rows; ++i)
      y[i] = target_data[start_row + i];
   int remaining = static_cast<int>(target_data.size()) - (start_row + num_rows);
   y_update = new double[remaining];
   for (int i = 0; i < remaining; ++i)
      y_update[i] = target_data[start_row + num_rows + i];
   int n_rows_mat = static_cast<int>(data_set.size()) - start_row;
   Eigen::MatrixXd mat(n_rows_mat, num_cols);
   for (int i = 0; i < n_rows_mat; ++i)
      for (int j = 0; j < num_cols; ++j)
         mat(i, j) = data_set[i + start_row][j];
   initial_matrix = mat.block(0, 0, num_rows, num_cols);
   update_matrix = mat.block(num_rows, 0, n_rows_mat - num_rows, num_cols);
}

struct RawSeries
{
   std::vector<double> x;
};

static inline RawSeries load_series()
{
   RawSeries s;
   std::vector<std::vector<std::string>> raw_data = read_csv_func("data/EURUSD/raw_norm_EURUSD.csv");
   for (size_t i = 1; i < raw_data.size(); ++i)
      s.x.push_back(std::stod(raw_data[i][0]));
   return s;
}

static inline void calculate_stats(const std::vector<double> &errors, double &mse, double &var)
{
   if (errors.empty())
   {
      mse = std::numeric_limits<double>::quiet_NaN();
      var = std::numeric_limits<double>::quiet_NaN();
      return;
   }
   double sum_sq = 0.0, sum = 0.0;
   for (double e : errors)
   {
      sum += e;
      sum_sq += (e * e);
   }
   mse = sum_sq / static_cast<double>(errors.size());
   double mean = sum / static_cast<double>(errors.size());
   if (errors.size() == 1)
   {
      var = 0.0;
      return;
   }
   double var_sum = 0.0;
   for (double e : errors)
      var_sum += std::pow(e - mean, 2);
   var = var_sum / static_cast<double>(errors.size() - 1);
}

#ifndef NDEBUG
template <typename Func>
static inline void debug_run(Func &&f)
{
   std::forward<Func>(f)();
}
#else
template <typename Func>
static inline void debug_run(Func &&)
{
}
#endif

static inline double check_weights(std::vector<std::vector<double>>& curr_X, double* curr_y, ABO &abo, GaussianRFF &g_rff)
{
   // Use the actual dimensions of the provided matrix `curr_X` rather than ABO internals
   int num_rows = static_cast<int>(curr_X.size());
   if (num_rows == 0)
      return 0.0;
   int num_cols = static_cast<int>(curr_X[0].size());
   int D = abo.dim_;

   Eigen::MatrixXd curr_X_eig(num_rows, num_cols);
   for (int i = 0; i < num_rows; ++i)
   {
      if (static_cast<int>(curr_X[i].size()) != num_cols)
         throw std::runtime_error("Inconsistent row sizes in curr_X");
      for (int j = 0; j < num_cols; ++j)
         curr_X_eig(i, j) = curr_X[i][j];
   }
   Eigen::MatrixXd X_to_check = g_rff.transform_matrix(curr_X_eig);

   std::vector<double> X_to_check_c_arr(static_cast<size_t>(D) * static_cast<size_t>(num_rows));
   for (int j = 0; j < D; ++j)
   {
      for (int i_1 = 0; i_1 < num_rows; ++i_1)
      {
         X_to_check_c_arr[static_cast<size_t>(i_1) + static_cast<size_t>(j) * static_cast<size_t>(num_rows)] = X_to_check(i_1, j);
      }
   }

   // QR decomposition — Q_local and R_temp have stride n_obs_
   double *Q_check;
   double *R_check;
   std::tie(Q_check, R_check) = Q_R_compute(X_to_check_c_arr.data(), num_rows, D);

   double *R_inv_check = new double[static_cast<size_t>(D) * static_cast<size_t>(num_rows)]();
   // Compute pseudo-inverse of R_temp into R_inv_ (stride dim_)
   pinv(R_check, R_inv_check, num_rows, D);

   // z = Q^T * y, then beta = R_inv * z
   std::vector<double> z_temp(static_cast<size_t>(num_rows));
   std::vector<double> beta_check(static_cast<size_t>(D));
   cblas_dgemv(CblasColMajor, CblasTrans,
               num_rows, num_rows, 1.0, Q_check, num_rows, curr_y, 1, 0.0, z_temp.data(), 1);
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               D, num_rows, 1.0, R_inv_check, D, z_temp.data(), 1, 0.0, beta_check.data(), 1);

   double diff = 0.0;
   for (int i_3 = 0; i_3 < D; i_3++)
   {
      double abo_b = abo.beta_[i_3];
      diff += (abo_b - beta_check[static_cast<size_t>(i_3)]) * (abo_b - beta_check[static_cast<size_t>(i_3)]);
   }

   delete[] Q_check;
   delete[] R_check;
   delete[] R_inv_check;

   // Return squared norm between ABO's beta and the direct QR solution
   return diff;
}

static inline double check_weights_qrd(const std::vector<std::vector<double>> &curr_X, double *curr_y, QRDRLS &qrd)
{
   int num_rows = static_cast<int>(curr_X.size());
   if (num_rows == 0)
      return 0.0;
   int num_cols = static_cast<int>(curr_X[0].size());

   Eigen::MatrixXd curr_X_eig(num_rows, num_cols);
   Eigen::VectorXd curr_y_eig(num_rows);
   for (int i = 0; i < num_rows; ++i)
   {
      if (static_cast<int>(curr_X[i].size()) != num_cols)
         throw std::runtime_error("Inconsistent row sizes in curr_X");
      for (int j = 0; j < num_cols; ++j)
      {
         curr_X_eig(i, j) = curr_X[i][j];
      }
      curr_y_eig(i) = curr_y[i];
   }

   Eigen::VectorXd beta_check = curr_X_eig.colPivHouseholderQr().solve(curr_y_eig);

   std::vector<double> beta_qrd(static_cast<size_t>(num_cols));
   qrd.getCoefficients(beta_qrd.data());

   double diff = 0.0;
   for (int j = 0; j < num_cols; ++j)
   {
      double delta = beta_qrd[j] - beta_check(j);
      diff += delta * delta;
   }

   return diff;
}

static inline double check_weights_krls(const std::vector<std::vector<double>> &curr_X, double *curr_y, KRLS_RBF &krls, double sigma, double regularizer)
{
   int n = static_cast<int>(curr_X.size());
   if (n == 0)
      return 0.0;
   int d = static_cast<int>(curr_X[0].size());

   std::vector<double> K(static_cast<size_t>(n) * n);
   for (int j = 0; j < n; ++j)
   {
      for (int i = 0; i < n; ++i)
      {
         double sum_sq = 0.0;
         for (int k = 0; k < d; ++k)
         {
            double diff = curr_X[i][k] - curr_X[j][k];
            sum_sq += diff * diff;
         }
         K[i + j * n] = std::exp(-sum_sq / (2.0 * sigma * sigma));
      }
   }

   std::vector<double> K_reg = K;
   for (int i = 0; i < n; ++i)
      K_reg[i + i * n] += regularizer;

   std::vector<double> P(static_cast<size_t>(n) * n);
   pinv(K_reg.data(), P.data(), n, n);

   std::vector<double> alpha(static_cast<size_t>(n));
   cblas_dgemv(CblasColMajor, CblasNoTrans,
               n, n, 1.0, P.data(), n, curr_y, 1, 0.0, alpha.data(), 1);

   double diff = 0.0;
   for (int i = 0; i < n; ++i)
   {
      double pred = 0.0;
      for (int j = 0; j < n; ++j)
         pred += alpha[j] * K[i + j * n];

      double model_pred = krls.predict(curr_X[i].data());
      diff += (pred - model_pred) * (pred - model_pred);
   }

   return diff;
}

// ---- per-model fold runners ----

static inline FoldResultRow run_fold_abo(
    const RawSeries &series, int first_date, int fold_k,
    int W, int L, double sigma, int D, int val_length,
    double ff, double regularizer)
{
   FoldResultRow row;
   row.fold = fold_k; row.model = "ABO"; row.L = L; row.W = W; row.sigma = sigma; row.D = D;

   std::vector<std::vector<double>> data_set;
   std::vector<double> target_data;
   lag_matrix(series.x, L, data_set, target_data);

   Eigen::MatrixXd initial_matrix, update_matrix;
   std::vector<double> y_vec(W);
   double *y_update = nullptr;

   const int start_row = first_date + val_length * fold_k;
   dataset_creation(data_set, target_data, initial_matrix, update_matrix, y_vec.data(), y_update, W, L, start_row);

   if (y_update == nullptr || update_matrix.rows() < val_length)
   {
      delete[] y_update;
      return row;
   }

   GaussianRFF g_rff(L, D, sigma, 0);
   Eigen::MatrixXd X_old = g_rff.transform_matrix(initial_matrix);
   std::vector<double> X_flat(W * D);
   for (int j = 0; j < D; ++j)
      for (int i = 0; i < W; ++i)
         X_flat[i + j * W] = X_old(i, j);

   ABO abo(X_flat.data(), y_vec.data(), W, ff, D, W);
   std::vector<std::vector<double>> X_raw_ring(W, std::vector<double>(L));
   //std::vector<double> y_raw_ring(W);
   double y_raw_ring[W];
   for (int ri = 0; ri < W; ri++){
      for (int j = 0; j < L; j++){
         X_raw_ring[ri][j] = initial_matrix(ri, j);
      }
      y_raw_ring[ri] = y_vec[ri];
   }
   int ring_idx = 0;

   std::vector<double> errors;
   errors.reserve(val_length);

   for (int i = 0; i < val_length; ++i)
   {
      Eigen::MatrixXd X_up_mat = g_rff.transform(update_matrix.row(i));

      if ((abo.n_obs_) == W)
      //if ((abo.n_obs_) == (W+1))
      {
         Eigen::MatrixXd raw_old_mat(1, L);
         for (int j = 0; j < L; j++)
            raw_old_mat(0, j) = X_raw_ring[ring_idx][j];
         Eigen::MatrixXd z_old_mat = g_rff.transform(raw_old_mat);
         std::vector<double> z_old_arr(D);
         for (int j = 0; j < D; j++)
            z_old_arr[j] = z_old_mat(0, j);
         abo.downdate(z_old_arr.data());
      }

      double pred = abo.pred(X_up_mat.data());
      errors.push_back(y_update[i] - pred);

      for (int j = 0; j < L; j++)
         X_raw_ring[ring_idx][j] = update_matrix(i, j);

      y_raw_ring[ring_idx] = y_update[i];

      abo.update(X_up_mat.data(), y_update[i]);

      debug_run([&]{
         double yes_no = check_weights(X_raw_ring, y_raw_ring, abo, g_rff);
         if (!std::isfinite(yes_no) || yes_no > 1e-8)
         {
            std::cout << fmt::format("[ABO compare] fold={} i={} weight_diff={}\n", fold_k, i, yes_no);
         }
      });

      ring_idx = (ring_idx + 1) % W;
      
      
   }

   calculate_stats(errors, row.mse, row.var);
   row.valid = true;

   delete[] y_update;
   return row;
}

static inline FoldResultRow run_fold_qrd(
    const RawSeries &series, int first_date, int fold_k,
    int W, int L, int val_length, double ff, double regularizer)
{
   FoldResultRow row;
   row.fold = fold_k; row.model = "QRD-RLS"; row.L = L; row.W = W; row.D = 0;

   std::vector<std::vector<double>> data_set;
   std::vector<double> target_data;
   lag_matrix(series.x, L, data_set, target_data);

   Eigen::MatrixXd initial_matrix, update_matrix;
   std::vector<double> y_vec(W);
   double *y_update = nullptr;

   const int start_row = first_date + val_length * fold_k;
   dataset_creation(data_set, target_data, initial_matrix, update_matrix, y_vec.data(), y_update, W, L, start_row);

   if (y_update == nullptr || update_matrix.rows() < val_length)
   {
      delete[] y_update;
      return row;
   }

   QRDRLS qrd(W, L, ff, regularizer);
   std::vector<double> X_flat(W * L);
   for (int j = 0; j < L; ++j)
      for (int i = 0; i < W; ++i)
         X_flat[i + j * W] = initial_matrix(i, j);
   qrd.batchInitialize(X_flat.data(), y_vec.data(), W, L);

   std::vector<std::vector<double>> X_raw_ring(W, std::vector<double>(L));
   double y_raw_ring[W];
   for (int ri = 0; ri < W; ++ri)
   {
      for (int j = 0; j < L; ++j)
         X_raw_ring[ri][j] = initial_matrix(ri, j);
      y_raw_ring[ri] = y_vec[ri];
   }
   int ring_idx = 0;

   std::vector<double> row_vec(L), errors;
   errors.reserve(val_length);

   for (int i = 0; i < val_length; ++i)
   {
      for (int j = 0; j < L; ++j)
         row_vec[j] = update_matrix(i, j);

      double p, e;
      qrd.update(row_vec.data(), y_update[i], p, e);
      errors.push_back(e);

      for (int j = 0; j < L; ++j)
         X_raw_ring[ring_idx][j] = update_matrix(i, j);
      y_raw_ring[ring_idx] = y_update[i];

      debug_run([&]{
         double yes_no = check_weights_qrd(X_raw_ring, y_raw_ring, qrd);
         if (!std::isfinite(yes_no) || yes_no > 1e-8)
         {
            std::cout << fmt::format("[QRD compare] fold={} i={} weight_diff={}\n", fold_k, i, yes_no);
         }
      });

      ring_idx = (ring_idx + 1) % W;
   }

   calculate_stats(errors, row.mse, row.var);
   row.valid = true;

   delete[] y_update;
   return row;
}

static inline FoldResultRow run_fold_krls(
    const RawSeries &series, int first_date, int fold_k,
    int W, int L, double sigma, int val_length, double ff, double regularizer)
{
   FoldResultRow row;
   row.fold = fold_k; row.model = "KRLS-RBF"; row.L = L; row.W = W; row.sigma = sigma; row.D = 0;

   std::vector<std::vector<double>> data_set;
   std::vector<double> target_data;
   lag_matrix(series.x, L, data_set, target_data);

   Eigen::MatrixXd initial_matrix, update_matrix;
   std::vector<double> y_vec(W);
   double *y_update = nullptr;

   const int start_row = first_date + val_length * fold_k;
   dataset_creation(data_set, target_data, initial_matrix, update_matrix, y_vec.data(), y_update, W, L, start_row);

   if (y_update == nullptr || update_matrix.rows() < val_length)
   {
      delete[] y_update;
      return row;
   }

   std::vector<double> X_flat(W * L);
   for (int j = 0; j < L; ++j)
      for (int i = 0; i < W; ++i)
         X_flat[i + j * W] = initial_matrix(i, j);
   KRLS_RBF krls(X_flat.data(), y_vec.data(), W, L, regularizer, sigma, W);

   std::vector<std::vector<double>> X_raw_ring(W, std::vector<double>(L));
   double y_raw_ring[W];
   for (int ri = 0; ri < W; ++ri)
   {
      for (int j = 0; j < L; ++j)
         X_raw_ring[ri][j] = initial_matrix(ri, j);
      y_raw_ring[ri] = y_vec[ri];
   }
   int ring_idx = 0;

   std::vector<double> row_vec(L), errors;
   errors.reserve(val_length);

   for (int i = 0; i < val_length; ++i)
   {
      for (int j = 0; j < L; ++j)
         row_vec[j] = update_matrix(i, j);

      double p, e;
      krls.update(row_vec.data(), y_update[i], p, e);
      errors.push_back(e);

      for (int j = 0; j < L; ++j)
         X_raw_ring[ring_idx][j] = update_matrix(i, j);
      y_raw_ring[ring_idx] = y_update[i];

      debug_run([&]{
         std::vector<std::vector<double>> X_ordered(W, std::vector<double>(L));
         std::vector<double> y_ordered(W);
         int start_idx = (ring_idx + 1) % W;
         for (int ri = 0; ri < W; ++ri)
         {
            int idx = (start_idx + ri) % W;
            X_ordered[ri] = X_raw_ring[idx];
            y_ordered[ri] = y_raw_ring[idx];
         }
         double yes_no = check_weights_krls(X_ordered, y_ordered.data(), krls, sigma, regularizer);
         if (!std::isfinite(yes_no) || yes_no > 1e-8)
         {
            std::cout << fmt::format("[KRLS compare] fold={} i={} pred_diff={}\n", fold_k, i, yes_no);
         }
      });

      ring_idx = (ring_idx + 1) % W;
   }

   calculate_stats(errors, row.mse, row.var);
   row.valid = true;

   delete[] y_update;
   return row;
}

// ---- CSV output ----

static inline void save_rows_csv(const std::vector<FoldResultRow> &rows, const std::string &path)
{
   std::ofstream f(path);
   if (!f)
   {
      std::cerr << "Error opening file: " << path << "\n";
      return;
   }

   f << "fold,model,L,W,sigma,D,mse,var\n";
   for (const auto &r : rows)
   {
      f << r.fold << "," << r.model << "," << r.L << "," << r.W << "," << r.sigma << "," << r.D << ","
        << r.mse << "," << r.var << "\n";
   }
}

// ---- main ----

int main(int argc, char **argv)
{
   Args args;
   try
   {
      parse_args(argc, argv, args);
   }
   catch (const std::exception &e)
   {
      std::cerr << "Arg error: " << e.what() << "\n\n";
      return 1;
   }

   RawSeries series = load_series();

   std::vector<int> folds(args.common.end_k - args.common.start_k);
   std::iota(folds.begin(), folds.end(), args.common.start_k);

   std::vector<FoldResultRow> rows;
   rows.reserve(static_cast<size_t>(folds.size()) * 3);

   struct Agg
   {
      int n = 0;
      double mse_sum = 0.0, var_sum = 0.0;
   };
   Agg agg_abo, agg_qrd, agg_krls;

   for (int k : folds)
   {
      const int fold_idx = k - args.common.start_k;

      if (args.common.run_abo)
      {
         std::cout << fmt::format("\n[ABO] fold={} L={} W={} sigma={} D={}\n", fold_idx, args.abo.L, args.abo.W, args.abo.sigma, args.abo.D);
         FoldResultRow r = run_fold_abo(series, args.common.first_date, k, args.abo.W, args.abo.L, args.abo.sigma, args.abo.D,
                                        args.common.val_length, args.abo.ff, args.abo.regularizer);
         if (r.valid)
         {
            rows.push_back(r);
            agg_abo.n++; agg_abo.mse_sum += r.mse; agg_abo.var_sum += r.var;
         }
         else
         {
            std::cout << fmt::format("[ABO] skipped invalid fold={}\n", fold_idx);
         }
      }

      if (args.common.run_qrd)
      {
         std::cout << fmt::format("\n[QRD-RLS] fold={} L={} W={}\n", fold_idx, args.qrd.L, args.qrd.W);
         FoldResultRow r = run_fold_qrd(series, args.common.first_date, k, args.qrd.W, args.qrd.L,
                                        args.common.val_length, args.qrd.ff, args.qrd.regularizer);
         if (r.valid)
         {
            rows.push_back(r);
            agg_qrd.n++; agg_qrd.mse_sum += r.mse; agg_qrd.var_sum += r.var;
         }
         else
         {
            std::cout << fmt::format("[QRD-RLS] skipped invalid fold={}\n", fold_idx);
         }
      }

      if (args.common.run_krls)
      {
         std::cout << fmt::format("\n[KRLS-RBF] fold={} L={} W={} sigma={}\n", fold_idx, args.krls.L, args.krls.W, args.krls.sigma);
         FoldResultRow r = run_fold_krls(series, args.common.first_date, k, args.krls.W, args.krls.L, args.krls.sigma,
                                         args.common.val_length, args.krls.ff, args.krls.regularizer);
         if (r.valid)
         {
            rows.push_back(r);
            agg_krls.n++; agg_krls.mse_sum += r.mse; agg_krls.var_sum += r.var;
         }
         else
         {
            std::cout << fmt::format("[KRLS-RBF] skipped invalid fold={}\n", fold_idx);
         }
      }
   }

   save_rows_csv(rows, args.common.out_csv);

   auto mean_or_nan = [](double s, int n) { return (n > 0) ? (s / n) : std::numeric_limits<double>::quiet_NaN(); };

   std::cout << "\n" << std::string(54, '=') << "\n";
   std::cout << fmt::format("{:<12} | {:<18} | {:<18}\n", "Method", "Mean MSE", "Mean Variance");
   std::cout << std::string(54, '-') << "\n";

   if (args.common.run_abo)
   {
      std::cout << fmt::format("{:<12} | {:<18.10f} | {:<18.10f}\n", 
                               "ABO", mean_or_nan(agg_abo.mse_sum, agg_abo.n), mean_or_nan(agg_abo.var_sum, agg_abo.n));
   }
   if (args.common.run_qrd)
   {
      std::cout << fmt::format("{:<12} | {:<18.10f} | {:<18.10f}\n", 
                               "QRD-RLS", mean_or_nan(agg_qrd.mse_sum, agg_qrd.n), mean_or_nan(agg_qrd.var_sum, agg_qrd.n));
   }
   if (args.common.run_krls)
   {
      std::cout << fmt::format("{:<12} | {:<18.10f} | {:<18.10f}\n", 
                               "KRLS-RBF", mean_or_nan(agg_krls.mse_sum, agg_krls.n), mean_or_nan(agg_krls.var_sum, agg_krls.n));
   }
   std::cout << std::string(54, '=') << std::endl;

   return 0;
}