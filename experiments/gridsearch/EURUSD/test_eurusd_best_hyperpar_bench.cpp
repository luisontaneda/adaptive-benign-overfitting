#include "abo/dd_test.h"
#include "baselines/QRD_RLS/qrd_rls.h"
#include "baselines/KRLS_RBF/krls_rbf.h"

#include <benchmark/benchmark.h>
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
    int end_k = 5;
    int val_length = 960 * 2;
    std::string out_csv = "results/gridsearch/EURUSD/best_test_bench.csv";
};

struct Args
{
    CommonParams common;
    ModelParamsABO abo;
    ModelParamsQRD qrd;
    ModelParamsKRLS krls;
};

struct Stats
{
    double mse = 0.0;
    double var = 0.0;
};

// ---- globals ----

static Args g_args;
static Stats g_abo_stats, g_qrd_stats, g_krls_stats;

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
        if (is_flag(argv[i], "--first_date"))
        {
            need("--first_date");
            a.common.first_date = std::stoi(argv[++i]);
        }
        else if (is_flag(argv[i], "--val_length"))
        {
            need("--val_length");
            a.common.val_length = std::stoi(argv[++i]);
        }
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
static RawSeries &get_series()
{
    static RawSeries s;
    static bool loaded = false;
    if (!loaded)
    {
        std::vector<std::vector<std::string>> raw_data = read_csv_func("data/EURUSD/raw_norm_EURUSD.csv");
        for (size_t i = 1; i < raw_data.size(); ++i)
            s.x.push_back(std::stod(raw_data[i][0]));
        loaded = true;
    }
    return s;
}

static Stats calculate_stats(const std::vector<double> &errors)
{
    if (errors.empty())
        return {0.0, 0.0};
    double sum_sq = 0.0, sum = 0.0;
    for (double e : errors)
    {
        sum += e;
        sum_sq += (e * e);
    }
    double mse = sum_sq / errors.size();
    double mean = sum / errors.size();
    double var_sum = 0.0;
    for (double e : errors)
        var_sum += std::pow(e - mean, 2);
    return {mse, var_sum / errors.size()};
}

// ---- Google Benchmark Functions ----

static void BM_ABO_Update(benchmark::State &state)
{
    const RawSeries &series = get_series();
    int L = g_args.abo.L, W = g_args.abo.W, D = g_args.abo.D;
    double sigma = g_args.abo.sigma, ff = g_args.abo.ff;
    int val_length = g_args.common.val_length;

    std::vector<std::vector<double>> data_set;
    std::vector<double> target_data;
    lag_matrix(series.x, L, data_set, target_data);

    Eigen::MatrixXd initial_matrix, update_matrix;
    std::vector<double> y_vec(W);
    double *y_update = nullptr;
    dataset_creation(data_set, target_data, initial_matrix, update_matrix, y_vec.data(), y_update, W, L, g_args.common.first_date);

    GaussianRFF g_rff(L, D, sigma, 0);
    Eigen::MatrixXd X_old = g_rff.transform_matrix(initial_matrix);
    std::vector<double> X_flat(W * D);
    for (int j = 0; j < D; ++j)
        for (int i = 0; i < W; ++i)
            X_flat[i + j * W] = X_old(i, j);

    ABO abo(X_flat.data(), y_vec.data(), W, ff, D, W);
    std::vector<std::vector<double>> X_raw_ring(W, std::vector<double>(L));
    for (int ri = 0; ri < W; ri++)
        for (int j = 0; j < L; j++)
            X_raw_ring[ri][j] = initial_matrix(ri, j);
    int ring_idx = 0;

    std::vector<double> errors;
    errors.reserve(val_length);

    for (auto _ : state)
    {
        errors.clear();
        for (int i = 0; i < val_length; ++i)
        {
            Eigen::MatrixXd X_up_mat = g_rff.transform(update_matrix.row(i));
            if (abo.n_obs_ == W)
            {
                Eigen::MatrixXd raw_old_mat(1, L);
                for (int j = 0; j < L; j++)
                    raw_old_mat(0, j) = X_raw_ring[ring_idx][j];
                Eigen::MatrixXd z_old_mat = g_rff.transform(raw_old_mat);
                abo.downdate(z_old_mat.data());
            }
            double pred = abo.pred(X_up_mat.data());
            errors.push_back(y_update[i] - pred);
            for (int j = 0; j < L; j++)
                X_raw_ring[ring_idx][j] = update_matrix(i, j);
            ring_idx = (ring_idx + 1) % W;
            abo.update(X_up_mat.data(), y_update[i]);
        }
    }
    g_abo_stats = calculate_stats(errors);
    state.SetItemsProcessed(state.iterations() * val_length);
    delete[] y_update;
}

static void BM_QRD_Update(benchmark::State &state)
{
    const RawSeries &series = get_series();
    int L = g_args.qrd.L, W = g_args.qrd.W, val_length = g_args.common.val_length;
    double ff = g_args.qrd.ff, reg = g_args.qrd.regularizer;

    std::vector<std::vector<double>> data_set;
    std::vector<double> target_data;
    lag_matrix(series.x, L, data_set, target_data);

    Eigen::MatrixXd initial_matrix, update_matrix;
    std::vector<double> y_vec(W);
    double *y_update = nullptr;
    dataset_creation(data_set, target_data, initial_matrix, update_matrix, y_vec.data(), y_update, W, L, g_args.common.first_date);

    QRDRLS qrd(W, L, ff, reg);
    std::vector<double> X_flat(W * L);
    for (int j = 0; j < L; ++j)
        for (int i = 0; i < W; ++i)
            X_flat[i + j * W] = initial_matrix(i, j);
    qrd.batchInitialize(X_flat.data(), y_vec.data(), W, L);

    std::vector<double> row_vec(L), errors;
    errors.reserve(val_length);

    for (auto _ : state)
    {
        errors.clear();
        for (int i = 0; i < val_length; ++i)
        {
            for (int j = 0; j < L; ++j)
                row_vec[j] = update_matrix(i, j);
            double p, e;
            qrd.update(row_vec.data(), y_update[i], p, e);
            errors.push_back(e);
        }
    }
    g_qrd_stats = calculate_stats(errors);
    state.SetItemsProcessed(state.iterations() * val_length);
    delete[] y_update;
}

static void BM_KRLS_Update(benchmark::State &state)
{
    const RawSeries &series = get_series();
    int L = g_args.krls.L, W = g_args.krls.W, val_length = g_args.common.val_length;
    double sigma = g_args.krls.sigma, reg = g_args.krls.regularizer;

    std::vector<std::vector<double>> data_set;
    std::vector<double> target_data;
    lag_matrix(series.x, L, data_set, target_data);

    Eigen::MatrixXd initial_matrix, update_matrix;
    std::vector<double> y_vec(W);
    double *y_update = nullptr;
    dataset_creation(data_set, target_data, initial_matrix, update_matrix, y_vec.data(), y_update, W, L, g_args.common.first_date);

    std::vector<double> X_flat(W * L);
    for (int j = 0; j < L; ++j)
        for (int i = 0; i < W; ++i)
            X_flat[i + j * W] = initial_matrix(i, j);
    KRLS_RBF krls(X_flat.data(), y_vec.data(), W, L, reg, 1.0 / sigma, W);

    std::vector<double> row_vec(L), errors;
    errors.reserve(val_length);

    for (auto _ : state)
    {
        errors.clear();
        for (int i = 0; i < val_length; ++i)
        {
            for (int j = 0; j < L; ++j)
                row_vec[j] = update_matrix(i, j);
            double p, e;
            krls.update(row_vec.data(), y_update[i], p, e);
            errors.push_back(e);
        }
    }
    g_krls_stats = calculate_stats(errors);
    state.SetItemsProcessed(state.iterations() * val_length);
    delete[] y_update;
}

BENCHMARK(BM_ABO_Update)->Unit(benchmark::kMillisecond)->Repetitions(1)->DisplayAggregatesOnly(true)->UseRealTime();
BENCHMARK(BM_QRD_Update)->Unit(benchmark::kMillisecond)->Repetitions(1)->DisplayAggregatesOnly(true)->UseRealTime();
BENCHMARK(BM_KRLS_Update)->Unit(benchmark::kMillisecond)->Repetitions(1)->DisplayAggregatesOnly(true)->UseRealTime();

int main(int argc, char **argv)
{
    try
    {
        parse_args(argc, argv, g_args);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Arg error: " << e.what() << "\n";
        return 1;
    }

    get_series();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    // Final Statistics Output
    std::cout << "\n"
              << std::string(60, '=') << "\n";
    std::cout << fmt::format("{:<15} | {:<20} | {:<20}\n", "Method", "MSE", "Variance");
    std::cout << std::string(60, '-') << "\n";
    std::cout << fmt::format("{:<15} | {:<20.10f} | {:<20.10f}\n", "ABO", g_abo_stats.mse, g_abo_stats.var);
    std::cout << fmt::format("{:<15} | {:<20.10f} | {:<20.10f}\n", "QRD-RLS", g_qrd_stats.mse, g_qrd_stats.var);
    std::cout << fmt::format("{:<15} | {:<20.10f} | {:<20.10f}\n", "KRLS-RBF", g_krls_stats.mse, g_krls_stats.var);
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}