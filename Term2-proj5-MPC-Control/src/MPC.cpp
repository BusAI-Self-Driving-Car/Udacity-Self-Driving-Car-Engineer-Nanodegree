#include <limits>
#include <math.h>

#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

size_t N = 10; // No. of timesteps in the horizon
double dt = 0.1; // Timestep duration

/* The solver takes all the state variables and actuator variables in a single
 * vector:
 * opt_vector =
 * [x0, ... xN-1, y0, ... yN-1,
 * psi0, ... psiN-1, v0, ... vN-1,
 * cte0, ... cteN-1,
 * epsi0,... epsiN-1,
 * delta0, ... deltaN-2,
 * a0, ... aN-2]
 */
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

/* This value assumes the model presented in the classroom is used.
 *
 * It was obtained by measuring the radius formed by running the vehicle in the
 * simulator around in a circle with a constant steering angle and velocity on a
 * flat terrain.
 *
 * Lf was tuned until the the radius formed by simulating the model
 * presented in the classroom matched the previous radius.
 *
 * Distance between vehicle center of mass and front axle
 */
const double Lf = 2.67;

class FG_eval {
private:
  const double ref_cte_ = 0.;
  const double ref_epsi_ = 0.;
  const double ref_v_ = 40.;

  // Weights used in the cost function
  const double w_cte_error_ = 2000.;
  const double w_epsi_error_ = 2000.;
  const double w_v_error_ = 100.;
  const double w_delta_ = 10.;
  const double w_a_ = 10.;
  const double w_change_delta_ = 100000;
  const double w_change_a_ = 10000;

public:
  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs_;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs_ = coeffs; }

  void operator()(ADvector& fg, const ADvector& opt_vector) {
    /*
    * `fg` a vector of the cost (fg[0]) and constraints (fg[1:end])
    * `opt_vector` is a vector of variable values (state & control inputs)
    */

    AD<double> &cost = fg[0];
    cost = 0.;

    for (size_t i = 0; i < N; i++) {
      // Minimize cross-track-error.
      cost += w_cte_error_ * CppAD::pow((opt_vector[cte_start+i] - ref_cte_), 2);

      // Minimize orientation error.
      cost += w_epsi_error_ * CppAD::pow((opt_vector[epsi_start+i] - ref_epsi_), 2);

      // Minimize deviation from reference velocity.
      cost += w_v_error_ * CppAD::pow((opt_vector[v_start+i] - ref_v_), 2);

      // Minimize actuation.
      if (i<(N-1)) {
        cost += w_delta_ * CppAD::pow(opt_vector[delta_start+i], 2);
        cost += w_a_ * CppAD::pow(opt_vector[a_start+i], 2);
      }

      // Minimize change in actuation (to prevent sudden movements).
      if (i<(N-2)) {
        cost += w_change_delta_ *
            CppAD::pow(opt_vector[delta_start+i+1] -
                       opt_vector[delta_start+i], 2);

        cost += w_change_a_ *
            CppAD::pow(opt_vector[a_start+i+1] - opt_vector[a_start+i], 2);
      }
    }

    // Setup initial model constraints
    fg[1 + x_start]     = opt_vector[x_start];
    fg[1 + y_start]     = opt_vector[y_start];
    fg[1 + psi_start]   = opt_vector[psi_start];
    fg[1 + v_start]     = opt_vector[v_start];
    fg[1 + cte_start]   = opt_vector[cte_start];
    fg[1 + epsi_start]  = opt_vector[epsi_start];

    // The rest of the constraints
    for (size_t i = 1; i < N; i++) {
      // time t+1
      AD<double> x1     = opt_vector[x_start + i];
      AD<double> y1     = opt_vector[y_start + i];
      AD<double> psi1   = opt_vector[psi_start + i];
      AD<double> v1     = opt_vector[v_start + i];
      AD<double> cte1   = opt_vector[cte_start + i];
      AD<double> epsi1  = opt_vector[epsi_start + i];

      // time t
      AD<double> x0     = opt_vector[x_start + i - 1];
      AD<double> y0     = opt_vector[y_start + i - 1];
      AD<double> psi0   = opt_vector[psi_start + i - 1];
      AD<double> v0     = opt_vector[v_start + i - 1];
      AD<double> cte0   = opt_vector[cte_start + i - 1];
      AD<double> epsi0  = opt_vector[epsi_start + i - 1];

      /* NOTE: don't know why delta0 has to be negated here for the program to
      work correctly! */
      AD<double> delta0 = -opt_vector[delta_start + i - 1];
      AD<double> a0     = opt_vector[a_start + i - 1];

      fg[1 + x_start + i] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + i] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);

      fg[1 + psi_start + i] = psi1 - (psi0 + v0/Lf * delta0 * dt);

      fg[1 + v_start + i] = v1 - (v0 + a0 * dt);

      // Evaluate third order polynomial at x = 0
      AD<double> fx_t = coeffs_[0] + coeffs_[1] * x0 + coeffs_[2] * x0 * x0 +
                        coeffs_[3] * x0 * x0 * x0;
      fg[1 + cte_start + i] = cte1 - (y0 - fx_t + v0 * CppAD::sin(epsi0) * dt);


      // Angle at x = 0: evaluate polynomial derivative at x = 0
      AD<double> psi_t = CppAD::atan(coeffs_[1] + 2 * coeffs_[2] * x0 +
                                     3 * coeffs_[3] * x0 * x0);
      fg[1 + epsi_start + i] = epsi1 - (psi0 - psi_t + v0/Lf * delta0 * dt);

    }
  }
};

MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {

  typedef CPPAD_TESTVECTOR(double) Dvector;

  /* TODO: Set the number of model variables (includes both states and inputs).
  * For example: If the state is a 4 element vector, the actuators is a 2
  * element vector and there are 10 timesteps. The number of variables is:
  * 4 * 10 + 2 * 9
  *
  * State: size = 6
  * [x, y -- position,
  * psi -- angle w.r.t. x-axis,
  * v -- actually, linear speed along psi,
  * cte -- cross-track-error,
  * epsi -- orientation error]
  *
  * Control: size = 2
  * [delta -- steering wheel angle -- +ve/-ve,
  * a -- throttle/brake -- brake -ve values, throttle +ve values]
  */
  size_t size = 6 * N + 2 * (N-1);

  auto x    = state[0];
  auto y    = state[1];
  auto psi  = state[2];
  auto v    = state[3];
  auto cte  = state[4];
  auto epsi = state[5];

  /* Initialize vector of variables to be optimized over (see opt_vector above).
  * opt_vector is structured as follows:
  * [x0, ... xN-1, y0, ... yN-1,
  * psi0, ... psiN-1, v0, ... vN-1,
  * cte0, ... cteN-1,
  * epsi0,... epsiN-1,
  * delta0, ... deltaN-2,
  * a0, ... aN-2]
  */
  Dvector opt_vector(size);
  for (size_t i = 0; i < size; i++) {
    opt_vector[i] = 0.0;
  }
  opt_vector[x_start]     = x;
  opt_vector[y_start]     = y;
  opt_vector[psi_start]   = psi;
  opt_vector[v_start]     = v;
  opt_vector[cte_start]   = cte;
  opt_vector[epsi_start]  = epsi;

  // Bounds on opt_vector -- state vector part.
  Dvector opt_vector_lowerbound(size);
  Dvector opt_vector_upperbound(size);
  for (size_t i = 0; i < delta_start; i++) {
    opt_vector_lowerbound[i] = -numeric_limits<float>::max();
    opt_vector_upperbound[i] = numeric_limits<float>::max();
  }

  // Bounds on opt_vector -- actuator variables part.
  // Steering angle [-25, 25], but expressed in radians
  double max_angle = 25.0 * M_PI / 180;
  for (size_t i = delta_start; i < a_start; i++) {
    opt_vector_lowerbound[i] = -max_angle;
    opt_vector_upperbound[i] = max_angle;
  }
  // Throttle
  for (size_t i = a_start; i < size; i++) {
    opt_vector_lowerbound[i] = -1.0;
    opt_vector_upperbound[i] = 1.0;
  }

  /* Constraints from the model-update equations.
   * e.g., xt+1 = xt + vt * cos(psit) * dt leads to the constraint:
   *       xt+1 - (xt + vt * cos(psit) * dt) = 0
   */
  size_t n_constraints = 6 * N;

  // Bounds on constraints.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (size_t i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[x_start]     = x;
  constraints_lowerbound[y_start]     = y;
  constraints_lowerbound[psi_start]   = psi;
  constraints_lowerbound[v_start]     = v;
  constraints_lowerbound[cte_start]   = cte;
  constraints_lowerbound[epsi_start]  = epsi;

  constraints_upperbound[x_start]     = x;
  constraints_upperbound[y_start]     = y;
  constraints_upperbound[psi_start]   = psi;
  constraints_upperbound[v_start]     = v;
  constraints_upperbound[cte_start]   = cte;
  constraints_upperbound[epsi_start]  = epsi;

  // options for IPOPT solver
  std::string options;

  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";

  /* NOTE: Setting sparse to true allows the solver to take advantage
  * of sparse routines, this makes the computation MUCH FASTER. If you
  * can uncomment 1 of these and see if it makes a difference or not but
  * if you uncomment both the computation time should go up in orders of
  * magnitude.
  */
  options += "Sparse  true        forward\n";
  //options += "Sparse  true        reverse\n";

  // NOTE: solver maximum time. Change if required.
  options += "Numeric max_cpu_time          0.5\n";

  FG_eval fg_eval(coeffs); // object that computes cost and constraints
  CppAD::ipopt::solve_result<Dvector> solution;
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options,
      opt_vector, opt_vector_lowerbound, opt_vector_upperbound,
      constraints_lowerbound, constraints_upperbound,
      fg_eval, solution);

  //assert(solution.status == CppAD::ipopt::solve_result<Dvector>::success);
  std::cout << "Cost " << solution.obj_value << std::endl << std::endl;

  // Return actuator values for the first timestep in the prediction horizon.
  vector<double> result;
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  // MPC prediction horizon
  for (size_t i = 0; i < N; i++) {
      result.push_back(solution.x[x_start + i]);
      result.push_back(solution.x[y_start + i]);
  }
  return result;
}
