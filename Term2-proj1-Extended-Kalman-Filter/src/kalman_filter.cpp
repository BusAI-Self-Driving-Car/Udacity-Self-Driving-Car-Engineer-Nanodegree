#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {
    // Initialize state covariance matrix P
    P_ = MatrixXd(4, 4).setZero();
    P_(2, 2) = 1000;
    P_(3, 3) = 1000;

    // Initial transition matrix F_
    F_ = MatrixXd(4, 4);
    F_ << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;

    // Initialize process noise covariance matrix
    Q_ = MatrixXd(4, 4).setZero();
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
    // KF Prediction step
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::UpdateLidar(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
    VectorXd y;
    y = z - H_ * x_;
    UpdateState(y);
}

void KalmanFilter::UpdateRadar(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
    VectorXd y(3), z_predicted;

    float px = x_(0), py = x_(1), vx = x_(2), vy = x_(3);
    z_predicted(0) = sqrt(px*px + py*py); // rho
    z_predicted(1) = atan2(py, px); // phi
    z_predicted(2) = (px * vx + py * vy) / std::max(0.00001, y(0)); // rho-dot, avoid division by zero

    y = z - z_predicted;

    // TODO ensure phi is within range [-pi, pi]
    while (y(1) > M_PI) y(1) -= 2 * M_PI;
    while (y(1) < -M_PI) y(1) += 2 * M_PI;

    UpdateState(y);
}

void KalmanFilter::UpdateState(const VectorXd &y) {
    MatrixXd S, K, H_t;
    MatrixXd I = MatrixXd::Identity(4, 4);

    H_t = H_.transpose();

    S = H_ * P_ * H_t + R_;

    // Kalman gain
    K = P_ * H_t * S.inverse();

    // KF Measurement update step
    x_ = x_ + K * y; // Mean
    P_ = (I - K * H_) * P_; // Covariance
}
