#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

size_t N = 10; // No. of timesteps in the horizon
double dt = 0.1; // Timestep duration

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
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    /* TODO: implement MPC
    * `fg` a vector of the cost constraints, `vars` is a vector of variable values
    * (state & control inputs)
    * NOTE: You'll probably go back and forth between this function and
    * the Solver function below.
    */
  }
};

MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  /* TODO: Set the number of model variables (includes both states and inputs).
  * For example: If the state is a 4 element vector, the actuators is a 2
  * element vector and there are 10 timesteps. The number of variables is:
  * 4 * 10 + 2 * 9
  *
  * State: [x, y -- position,
  *         psi -- angle w.r.t. x-axis,
  *         v -- actually, linear speed along psi]
  *
  * Control: [delta -- steering wheel angle -- +ve/-ve,
  *           a -- throttle/brake -- brake -ve values, throttle +ve values]
  */
  size_t n_vars = 4 * N + 2 * (N-1);

  /* TODO: Set the number of constraints
   * delta: [-1, 1]
   * throttle: [-1, 1]
   */
  size_t n_constraints = 2 * N;

  // Initial value of the independent variables.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

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

  CppAD::ipopt::solve_result<Dvector> solution;
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options,
      vars, vars_lowerbound, vars_upperbound,
      constraints_lowerbound, constraints_upperbound,
      fg_eval, solution);

  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  /* TODO: Return the first actuator values. The variables can be accessed with
  * `solution.x[i]`.
  *
  * {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  * creates a 2 element double vector.
  */
  return {};
}
