#include "gtest/gtest.h"

#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"

using namespace std;
using namespace sdcnd;

constexpr float threshold_matrix_diff = 1e-4;

double getMatrixDiffScalar(bool verbose, const MatrixXd& a, const MatrixXd& b)
{
    auto diff_scalar = (a - b).norm();
    if (verbose && diff_scalar < 0.1)
        std::cout << "[INFO] getMatrixDiffScalar(): Matrices approx. equal, diff_scalar: " << diff_scalar << std::endl;

    return diff_scalar;
}


TEST(ukfTest, test_CalcAugmentedSigmaPoints)
{
    UKF ukf;

    ukf.std_a_ = 0.2;
    ukf.std_yawdd_ = 0.2;

    //set example state
    VectorXd x = VectorXd(5);
    x <<   5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;
    ukf.x_ = x;

    //create example covariance matrix
    MatrixXd P = MatrixXd(5, 5);
    P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
    ukf.P_ = P;

    MatrixXd Xsig_aug(7, 15);
    Xsig_aug <<
      5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
        1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
      2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
      0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
      0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
           0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0,
           0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641;

    ukf.CalcAugmentedSigmaPoints();
    //std::cout << "ukf.Xsig_aug_: " << ukf.Xsig_aug_ << std::endl;

    auto diff_scalar = getMatrixDiffScalar(false, ukf.Xsig_aug_, Xsig_aug);
    ASSERT_LT(diff_scalar, threshold_matrix_diff);
}

TEST(ukfTest, test_PredictSigmaPoints)
{
    UKF ukf;

    double delta_t = 0.1; //time diff in sec

    ukf.Xsig_aug_ <<
       5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
         1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
       2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
       0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
       0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
            0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
            0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;


    MatrixXd Xsig_pred(5, 15);
    Xsig_pred <<
    5.93553, 6.06251, 5.92217, 5.9415, 5.92361, 5.93516, 5.93705, 5.93553, 5.80832, 5.94481, 5.92935, 5.94553, 5.93589, 5.93401, 5.93553,
    1.48939, 1.44673, 1.66484, 1.49719, 1.508, 1.49001, 1.49022, 1.48939, 1.5308, 1.31287, 1.48182, 1.46967, 1.48876, 1.48855, 1.48939,
    2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.23954, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.17026, 2.2049,
    0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
    0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.387441, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.318159;

    ukf.PredictStateSigmaPoints(delta_t);
    //std::cout << "ukf.Xsig_pred_: " << ukf.Xsig_pred_ << std::endl;

    auto diff_scalar = getMatrixDiffScalar(false, ukf.Xsig_pred_, Xsig_pred);
    ASSERT_LT(diff_scalar, threshold_matrix_diff);


}

TEST(ukfTest, test_CalcPredictedStateAndCovariance)
{
    UKF ukf;

    ukf.Xsig_pred_ <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

      VectorXd x(5);
      x <<  5.93637,
            1.49035,
            2.20528,
            0.536853,
            0.353577;

      MatrixXd P(5, 5);
      P <<  0.00543425, -0.0024053, 0.00341576, -0.00348196, -0.00299378,
            -0.0024053, 0.010845, 0.0014923, 0.00980182, 0.00791091,
            0.00341576, 0.0014923, 0.00580129, 0.000778632, 0.000792973,
            -0.00348196, 0.00980182, 0.000778632, 0.0119238, 0.0112491,
            -0.00299378, 0.00791091, 0.000792973, 0.0112491, 0.0126972;

      ukf.CalcPredictedStateAndCovariance();
      //std::cout << "ukf.x_: " << ukf.x_ << std::endl;
      //std::cout << "ukf.P_: " << ukf.P_ << std::endl;

      auto diff_scalar = getMatrixDiffScalar(false, ukf.x_, x);
      ASSERT_LT(diff_scalar, threshold_matrix_diff);

      diff_scalar = getMatrixDiffScalar(false, ukf.P_, P);
      ASSERT_LT(diff_scalar, threshold_matrix_diff);
}

TEST(ukfTest, test_PredictRadarMeasurement)
{
    UKF ukf;

    //radar measurement noise standard deviation radius in m
    ukf.std_radr_ = 0.3;

    //radar measurement noise standard deviation angle in rad
    ukf.std_radphi_ = 0.0175;

    //radar measurement noise standard deviation radius change in m/s
    ukf.std_radrd_ = 0.1;

    ukf.Xsig_pred_ <<
      5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
         1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
        2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
       0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
        0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

    VectorXd z_pred_radar(3, 1);
    z_pred_radar <<  6.12155,
                     0.245993,
                     2.10313;

    MatrixXd S_pred_radar(3, 3);
    S_pred_radar <<  0.0946171, -0.000139448, 0.00407016,
                     -0.000139448, 0.000617548, -0.000770652,
                     0.00407016, -0.000770652, 0.0180917;

    ukf.PredictRadarMeasurement();
    //std::cout << "ukf.z_pred_radar_: " << std::endl << ukf.z_pred_radar_ << std::endl;
    //std::cout << "ukf.S_pred_radar_: " << std::endl << ukf.S_pred_radar_ << std::endl;

    auto diff_scalar = getMatrixDiffScalar(false, ukf.z_pred_radar_, z_pred_radar);
    ASSERT_LT(diff_scalar, threshold_matrix_diff);

    diff_scalar = getMatrixDiffScalar(false, ukf.S_pred_radar_, S_pred_radar);
    ASSERT_LT(diff_scalar, threshold_matrix_diff);
}

TEST(ukfTest, test_UpdateStateAndCovarianceFromRadarMsmt)
{
    UKF ukf;

    ukf.Xsig_pred_ <<
       5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
         1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
        2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
       0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
        0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

    ukf.x_ <<
          5.93637,
           1.49035,
           2.20528,
          0.536853,
          0.353577;

    ukf.P_ <<
        0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
       -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
        0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
       -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
       -0.0029937,  0.0079109, 0.00079297,   0.011249,   0.0126972;

    ukf.Zsig_pred_radar_ <<
        6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
        0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
        2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;

    ukf.z_pred_radar_ <<
         6.12155,
         0.245993,
          2.10313;

    ukf.S_pred_radar_ <<
         0.0946171, -0.000139448,   0.00407016,
        -0.000139448,  0.000617548, -0.000770652,
         0.00407016, -0.000770652,    0.0180917;

    MeasurementPackage meas_package;
    meas_package.sensor_type_ = MeasurementPackage::RADAR;
    meas_package.raw_measurements_ = VectorXd(3);
    meas_package.raw_measurements_ <<
                                      5.9214,
                                      0.2187,
                                      2.0062;

    VectorXd x(ukf.n_x_);
    x <<
         5.92276,
         1.41823,
         2.15593,
        0.489274,
        0.321338;

     MatrixXd P(ukf.n_x_, ukf.n_x_);
     P <<
      0.00361579, -0.000357881,   0.00208316, -0.000937196,  -0.00071727,
    -0.000357881,   0.00539867,   0.00156846,   0.00455342,   0.00358885,
      0.00208316,   0.00156846,   0.00410651,   0.00160333,   0.00171811,
    -0.000937196,   0.00455342,   0.00160333,   0.00652634,   0.00669436,
     -0.00071719,   0.00358884,   0.00171811,   0.00669426,   0.00881797;

     ukf.UpdateStateAndCovarianceFromRadarMsmt(meas_package);

     //std::cout << "ukf.x_: " << std::endl << ukf.x_ << std::endl;
     //std::cout << "ukf.P_: " << std::endl << ukf.P_ << std::endl;

     auto diff_scalar = getMatrixDiffScalar(false, ukf.x_, x);
     ASSERT_LT(diff_scalar, threshold_matrix_diff);

     diff_scalar = getMatrixDiffScalar(false, ukf.P_, P);
     ASSERT_LT(diff_scalar, threshold_matrix_diff);

}

TEST(ukfTest, DISABLED_test_PredictLidarMeasurement)
{
    UKF ukf;
    ASSERT_TRUE(false);
}

TEST(ukfTest, DISABLED_test_UpdateStateAndCovarianceFromLidarMsmt)
{
    UKF ukf;
    ASSERT_TRUE(false);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return (RUN_ALL_TESTS());
}
