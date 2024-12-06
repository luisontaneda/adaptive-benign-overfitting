#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>
#include "read_csv_func.cpp"
// #include "QR_RLS.cpp"
#include "QR_RLS.h"
#include "gau_rff.h"

using namespace std;

Eigen::MatrixXd lag_matrix(const std::vector<double> &x, int lag)
{
    int n = x.size();
    int num_rows = n - lag + 1;

    // Initialize the output matrix with `num_rows` rows and `lag` columns
    Eigen::MatrixXd result(num_rows, lag);

    // Fill the matrix with lagged values
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = 0; j < lag; ++j)
        {
            result(i, j) = x[i + j];
        }
    }

    return result;
}

int main()
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

    vector<vector<string>> data_set;
    data_set = read_csv_func("daily_vix.csv");
    vector<double> close_price;
    vector<double> ret_price;

    for (int i = 2; i < data_set.size(); i++)
    // for (auto i :: )
    {
        // cout << data_set[i][4] << endl;
        double temp_num = stod(data_set[i][4]);
        close_price.push_back(temp_num);

        // double ret = stod(data_set[i][4]) / stod(data_set[i - 1][4]) - 1;
        double ret = log(stod(data_set[i][4]) / stod(data_set[i - 1][4]));
        ret_price.push_back(ret);
    }

    int n_lags = 14;
    MatrixXd close_lag_mat = lag_matrix(ret_price, n_lags);

    int start_row = 0;
    int start_col = 0;
    int num_rows = 20;
    int num_cols = n_lags;

    int num_elements = close_price.size() - n_lags - num_rows;
    Eigen::Map<Eigen::VectorXd> y(close_price.data() + n_lags + 1, num_rows);
    Eigen::Map<Eigen::VectorXd> y_update(close_price.data() + n_lags + num_rows + 1, num_elements);

    // cout << y.row(y.rows() - 1) << endl;
    // cout << y_update.row(0) << endl;

    MatrixXd initial_matrix = close_lag_mat.block(0, 0, num_rows, num_cols);
    MatrixXd update_matrix = close_lag_mat.block(num_rows, 0, close_lag_mat.rows() - num_rows, num_cols);

    // cout << initial_matrix.row(initial_matrix.rows() - 1) << endl;
    // cout << update_matrix.row(0) << endl;

    int d = num_cols;
    int D = 1000;
    double kernel_var = 1.0;
    bool seed = true;

    GaussianRFF g_rff(d, D, kernel_var, seed);

    MatrixXd X = g_rff.transform_matrix(initial_matrix);

    cout << X.rows() << " " << X.cols() << endl;
    // cout << X << endl;

    int max_obs = 25;
    double ff = 1;
    double lambda = 0.1;
    QR_Rls qr_rls(X, y, max_obs, ff, lambda);
    // QR_Rls qr_rls(initial_matrix, y, max_obs, ff, lambda);

    vector<double> preds;
    vector<double> mse;

    // MatrixXd X_update = g_rff.transform(update_matrix.row(0));
    //  cout << X_update.row(0) << endl;
    //  preds.push_back(qr_rls.pred(X_update));
    cout << initial_matrix.row(0) << endl;

    int n_its = 400;
    for (int i = 0; i < n_its; i++)
    {
        MatrixXd X_update = g_rff.transform(update_matrix.row(i));
        qr_rls.update(X_update, y_update.row(i));
        // qr_rls.update(update_matrix.row(i), y_update.row(i));
        cout << y_update.row(i) << endl;

        preds.push_back(qr_rls.pred(X_update));
        // preds.push_back(qr_rls.pred(update_matrix.row(i)));
        mse.push_back(pow(preds[i] - y_update.row(i)(0), 2));
    }
    // preds = [mod.pred(rff.transform(norm_exog[:,batch_s-1].reshape(lags,1)))]
    // mse = [(preds[0]-endog[0,batch_s])**2]

    // for i in range(1,n):
    //     u = rff.transform(norm_exog[:,batch_s+i-1].reshape(lags,1)) //# reshape feature vector to (4,1)
    //     d = endog[0,batch_s+i-1]
    //     mod.update(u,d)
    //    preds.append(mod.pred(rff.transform(norm_exog[:,batch_s+i-1].reshape(lags,1))))
    //    mse.append((preds[i]-endog[0,batch_s+i-1])**2)

    // std::cout << data_set << std::endl;

    //

    // qr_rls.update(up_x.row(0), u_y.row(0));

    return 0;
};