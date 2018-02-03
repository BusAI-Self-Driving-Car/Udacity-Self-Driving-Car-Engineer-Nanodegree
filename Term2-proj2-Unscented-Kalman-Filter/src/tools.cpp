#include <iostream>
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
