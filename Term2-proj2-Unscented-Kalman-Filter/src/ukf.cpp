#include "ukf.h"
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(0);
  weights_(0) = lambda_/(lambda_ + n_aug_);

  for(size_t i = 1; i < 2*n_aug_+1; ++i)
  {
    weights_(i) = 0.5/(lambda_ + n_aug_);
  }

  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  n_z_lidar_ = 2;
  z_pred_lidar_ = VectorXd(n_z_lidar_);
  S_pred_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  Zsig_pred_lidar_ = MatrixXd(n_z_lidar_, 2*n_aug_+1);

  n_z_radar_ = 3;
  z_pred_radar_ = VectorXd(n_z_radar_);
  S_pred_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  Zsig_pred_radar_ = MatrixXd(n_z_radar_, 2*n_aug_+1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // DO NOT MODIFY measurement noise values below these are provided by the
  // sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  // DO NOT MODIFY measurement noise values above these are provided by the
  // sensor manufacturer.

  /**
  TODO: Complete ctor

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO: ProcessMeasurement

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  static long previous_timestamp;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        x_ << meas_package.raw_measurements_(0),
            meas_package.raw_measurements_(1),
            0., 0., 0.;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        /**
        Convert radar from polar to cartesian coordinates and initialize state.
        */
        float rho     = meas_package.raw_measurements_(0);
        float theta   = meas_package.raw_measurements_(1);
        float rho_dot = meas_package.raw_measurements_(2);

        x_ << rho * cos(theta),    /* px */
                 rho * sin(theta),    /* py */
                 rho_dot,   // assume tracked obj moving in a straight line away from sensor
                 0., 0.;

    }

    is_initialized_ = true;
    previous_timestamp = meas_package.timestamp_;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  double delta_t = (meas_package.timestamp_ - previous_timestamp)/(double)1e6;
  PredictState(delta_t);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    PredictLidarMeasurement();
    UpdateStateAndCovarianceFromLidarMsmt(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    PredictRadarMeasurement();
    UpdateStateAndCovarianceFromRadarMsmt(meas_package);
  }

}

void UKF::CalcAugmentedSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; ++i)
  {
    Xsig_aug_.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  return;
}

void UKF::PredictStateSigmaPoints(double delta_t) {

  for(size_t col = 0; col < Xsig_aug_.cols(); ++col)
  {
    auto px = Xsig_aug_(0, col);
    auto py = Xsig_aug_(1, col);

    auto v  = Xsig_aug_(2, col);

    auto psi   = Xsig_aug_(3, col);
    auto psi_d = Xsig_aug_(4, col);

    auto nu_a      = Xsig_aug_(5, col);
    auto nu_psi_dd = Xsig_aug_(6, col);

    if(abs(psi_d) < 0.001) {
      Xsig_pred_(0, col) = px + (v * cos(psi) * delta_t);
      Xsig_pred_(1, col) = py + (v * sin(psi) * delta_t);
    } else {
      Xsig_pred_(0, col) = px + (v * (sin(psi + delta_t*psi_d) - sin(psi))) / psi_d;
      Xsig_pred_(1, col) = py + (v * (-cos(psi + delta_t*psi_d) + cos(psi))) / psi_d;
    }

    Xsig_pred_(0, col) += 0.5 * delta_t*delta_t * cos(psi) * nu_a;
    Xsig_pred_(1, col) += 0.5 * delta_t*delta_t * sin(psi) * nu_a;

    Xsig_pred_(2, col) = v + delta_t * nu_a;

    Xsig_pred_(3, col) = psi + delta_t * psi_d +
                          0.5 * delta_t*delta_t * nu_psi_dd;

    Xsig_pred_(4, col) = psi_d + delta_t * nu_psi_dd;

  }
}

void UKF::CalcPredictedStateAndCovariance() {

    x_.fill(0.0);
    std::cout << "Xsig_pred_.cols(): " << Xsig_pred_.cols() << std::endl;
    for (size_t i = 0; i < Xsig_pred_.cols(); ++i)
    {
        // Predict state
        //std::cout << weights_(i) << std::endl;
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    VectorXd diff;
    P_.fill(0.0);
    for (size_t i = 1; i < Xsig_pred_.cols(); ++i)
    {
        diff = (Xsig_pred_.col(i) - x_);

        //angle normalization
        while (diff(3)> M_PI) diff(3)-=2.*M_PI;
        while (diff(3)<-M_PI) diff(3)+=2.*M_PI;

        // Predict state covariance matrix
        P_ += weights_(i) * diff * diff.transpose();
    }
}

void UKF::PredictState(double delta_t) {
  /**
  TODO: Prediction

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  CalcAugmentedSigmaPoints();

  PredictStateSigmaPoints(delta_t);

  CalcPredictedStateAndCovariance();
}

void UKF::PredictLidarMeasurement() {
  // Xsig_pred_ --> Zsig_pred_lidar_
  // Zsig_pred_lidar_ --> z_pred_lidar_, S_pred_lidar_
}

void UKF::UpdateStateAndCovarianceFromLidarMsmt(MeasurementPackage meas_package) {
  /**
  TODO: UpdateLidar

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // Cross-correlation between sigma points from state-space and measurement-space
  // Zsig_pred_lidar_, Xsig_pred_, z_pred_lidar, x_ --> T_cross_corr_state_msmt_

  // Kalman gain
  // T_cross_corr_state_msmt_, S_pred_lidar_ --> K_

  // State update
  // x_, K_, z_lidar_, z_pred_lidar_  --> x_

  // Covariance update
  // P_, K_, S_pred_lidar_ --> P_

}

void UKF::PredictRadarMeasurement() {

  /*****************************************************************************
   *  Transform sigma points from state space to measurement space
   ****************************************************************************/
  // Xsig_pred_ [5x15] --> Zsig_pred_radar_ [3x15]
  for(size_t i = 0; i < Xsig_pred_.cols(); ++i)
  {
    auto px   = Xsig_pred_(0, i);
    auto py   = Xsig_pred_(1, i);
    auto v    = Xsig_pred_(2, i);
    auto psi  = Xsig_pred_(3, i);

    auto vx = v * cos(psi);
    auto vy = v * sin(psi);

    Zsig_pred_radar_(0, i) = sqrt(px*px + py*py); // rho
    Zsig_pred_radar_(1, i) = atan2(py, px); // phi
    Zsig_pred_radar_(2, i) = (px * vx + py * vy) / std::max(0.00001, Zsig_pred_radar_(0, i)); // rho-dot, avoid division by zero
  }

  /*****************************************************************************
   * Calculate predicted measurement mean and covariance
   ****************************************************************************/
  // Mean
  // Zsig_pred_radar_ [3x15] --> z_pred_radar_ [3x1]
  z_pred_radar_.fill(0.0);
  for (size_t i = 0; i < Zsig_pred_radar_.cols(); ++i)
  {
      z_pred_radar_ += weights_(i) * Zsig_pred_radar_.col(i);
  }

  // Innovation covariance matrix
  // Zsig_pred_radar_ [3x15] --> S_pred_radar_ [3x3]
  VectorXd diff;
  S_pred_radar_.fill(0.0);
  for (size_t i = 1; i < Zsig_pred_radar_.cols(); ++i)
  {
      diff = (Zsig_pred_radar_.col(i) - z_pred_radar_);

      // Psi normalization
      while (diff(1)> M_PI) diff(1)-=2.*M_PI;
      while (diff(1)<-M_PI) diff(1)+=2.*M_PI;

      S_pred_radar_ += weights_(i) * diff * diff.transpose();
  }

  // Add measurement noise covariance
  S_pred_radar_(0, 0) += std_radr_*std_radr_;
  S_pred_radar_(1, 1) += std_radphi_*std_radphi_;
  S_pred_radar_(2, 2) += std_radrd_*std_radrd_;
}

void UKF::UpdateStateAndCovarianceFromRadarMsmt(MeasurementPackage meas_package) {
  /**
  TODO: UpdateRadar

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // Cross-correlation between sigma points from state-space and measurement-space
  // Zsig_pred_radar_, Xsig_pred_, z_pred_radar, x_ --> T_cross_corr_state_msmt_

  // Kalman gain
  // T_cross_corr_state_msmt_, S_pred_radar_ --> K_

  // State update
  // x_, K_, z_radar_, z_pred_radar_  --> x_

  // Covariance update
  // P_, K_, S_pred_radar_ --> P_
}
