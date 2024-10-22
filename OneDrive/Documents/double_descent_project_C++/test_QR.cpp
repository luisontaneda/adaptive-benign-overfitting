#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>
// #include "QR_RLS.cpp"
#include "QR_RLS.h"

int main()
{
    typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> MatrixXld;
    typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> VectorXld;

    // Eigen::MatrixXd x(2, 3);
    // x << 1, 2, 3,
    //     4, 5, 6;
    MatrixXld x(2, 10);
    x << 1.0L, 2.0L, 3.0L, 4.0L, 5.0L, 6.0L, 7.0L, 8.0L, 9.0L, 10.0L,
        11.0L, 12.0L, 13.0L, 14.0L, 15.0L, 16.0L, 17.0L, 18.0L, 19.0L, 20.0L;
    VectorXld y(2);
    y << 1.0L, 2.0L;

    MatrixXld up_x(2, 10);
    up_x << 20.0L, 21.0L, 22.0L, 23.0L, 24.0L, 25.0L, 26.0L, 27.0L, 28.0L, 29.0L,
        30.0L, 31.0L, 32.0L, 33.0L, 34.0L, 35.0L, 36.0L, 37.0L, 38.0L, 39.0L;
    VectorXld u_y(2);
    u_y << 3.0L, 4.0L;

    std::cout << "Type of myMatrix: " << typeid(x).name() << std::endl;

    int max_obs = 10;
    long double ff = 1;
    long double lambda = 0.1;
    QR_Rls qr_rls(x, y, max_obs, ff, lambda);

    qr_rls.update(up_x.row(0), u_y.row(0));

    return 0;
};