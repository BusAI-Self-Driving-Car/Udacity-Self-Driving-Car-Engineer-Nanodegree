#include <iostream>
#include <stdexcept>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	VectorXd diff(4), rmse(4);
	diff << 0,0,0,0;
	rmse << 0,0,0,0;

	if (estimations.size() != ground_truth.size())
	{
		throw std::runtime_error ("Input vectors must have the same size!");
	}
	else if(estimations.size() == 0)
	{
		throw std::runtime_error ("Input vectors have zero length!");
	}

	for(std::size_t i = 0; i < estimations.size(); ++i)
	{
		diff = estimations[i] - ground_truth[i];
		diff = diff.array() * diff.array();

		rmse += diff;
	}

	rmse = rmse/estimations.size();
	return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
	MatrixXd H_j(3, 4);

	float px = x_state(0), py = x_state(1), vx = x_state(2), vy = x_state(3);

	float ms = px*px + py*py;
	float rms = sqrt(ms);
	float rms_cubed = ms*rms;

	// Check for division by zero
	if (rms < std::numeric_limits<double>::epsilon())
	{
		throw std::runtime_error ("Division by zero!");
	}

	H_j << px/rms, py/rms, 0, 0,
	      -py/ms, px/ms, 0, 0,
	      py*(vx*py - vy*px)/rms_cubed, px*(vy*px - vx*py)/rms_cubed, px/rms, py/rms;

	return H_j;
}
