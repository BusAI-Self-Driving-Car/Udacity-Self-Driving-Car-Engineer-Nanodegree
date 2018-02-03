#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace sdcnd {

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* Augmented sigma points matrix
  MatrixXd Xsig_aug_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  VectorXd z_pred_lidar_;
  MatrixXd S_pred_lidar_;
  MatrixXd Zsig_pred_lidar_;
  double NIS_lidar_;

  VectorXd z_pred_radar_;
  MatrixXd S_pred_radar_;
  MatrixXd Zsig_pred_radar_;
  double NIS_radar_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  int n_z_lidar_;
  int n_z_radar_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;


  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  void CalcAugmentedSigmaPoints();
  void PredictStateSigmaPoints(double delta_t);
  void CalcPredictedStateAndCovariance();

  /**
   * ProcessMeasurement
   * @param {MeasurementPackage} meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
   */
  void PredictState(double delta_t);

  /**
   * Predicts the Lidar measurement mean and covariance matrix
   */
  void PredictLidarMeasurement();

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param {MeasurementPackage} meas_package The measurement at k+1
   */
  void UpdateStateAndCovarianceFromLidarMsmt(MeasurementPackage meas_package);

  /**
   * Predicts the Radar measurement mean and covariance matrix
   */
  void PredictRadarMeasurement();

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param {MeasurementPackage} meas_package The measurement at k+1
   */
  void UpdateStateAndCovarianceFromRadarMsmt(MeasurementPackage meas_package);
};

} // namespace sdcnd

#endif /* UKF_H */
