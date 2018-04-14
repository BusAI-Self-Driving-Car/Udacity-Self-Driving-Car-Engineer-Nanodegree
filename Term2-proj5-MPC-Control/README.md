# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

<p align="center">
 <a href=""><img src="./videos/MPC.gif" alt="Overview" width="50%" height="50%"></a>
</p>

### Model-predictive control
Model Predictive Control is used in this project to generate actuator commands for a car in the Udacity simulator. From the simulator, it takes a reference trajectory to be followed as input in the form of waypoints corresponding to the road-center. It then optimizes vehicle state and actuator commands over a certain prediction time horizon to follow the reference trajectory as closely as possible. Since it explicitly models vehicle state, it is able to compensate for latency between actuator command and response by incorporating the (known) latency in its model.

#### Vehicle state
The vehicle state is defined as follows:
```python
[x, y -- position,
psi -- angle w.r.t. x-axis,
v -- actually, linear speed along psi,
cte -- cross-track-error,
epsi -- orientation error]
```
#### Actuator commands
The commands consist of the steering angle delta with bounds [-25°, 25°] and the throttle parameter [-1, 1].

#### Cost function for optimizer
The cost function is devised to achieve the following goals and is implemened in lines 76--98 in `MPC.cpp`:
* Minimize cross-track-error.
* Minimize orientation error.
* Minimize deviation from reference velocity.
* Minimize actuation.
* Minimize change in actuation (to prevent sudden movements).

#### Kinematic model
The kinematic model used for state update is as follows. The equations below lead to constraints for the optimizer, while it tries to optimize the cost function.

```python
x_t+1    = x_t    + v_t * cos(psi_t) * dt
y_t+1    = y_t    + v_t * sin(psi_t) * dt
psi_t+1  = psi_t  + v_t * delta_t/Lf * dt
v_t+1    = v_t    + acc_t * dt
cte_t+1  = cte_t  + v_t * sin(epsi_t) * dt
epsi_t+1 = epsi_t + v_t * delta_t/Lf * dt
```

### Parameters N and dt
N -- number of timesteps in prediction horizon
dt --duration of each timestep

The very first combination I tried was the one suggested in the Udacity classroom:
```python
size_t N = 25 ;
double dt = 0.05 ;
```

The result was that after a while, the car started oscillating about the center of the road in the simulator. This combination is computationally quite complex, as for every call to `mpc.Solve()`, the program minimizes the cost function over `6 * N + 2 * (N-1) = 198` variables. This slows down the actuator command rate leading to instability of the car. Also, the timestep resolution of `0.05` could perhaps also be increased as not much changes in the motion of the car over 50 ms.

The next combination I tried (and which worked well) was the following:
```python
size_t N = 10 ;
double dt = 0.1 ;
```
Here, we optimize over only 78 variables per iteration. The timestep resolution is also doubled. The duration of the prediction horizon is thus `10*0.1 = 1s` which is a reasonable time frame for the prediction. Anything longer is perhaps not useful as the road may change significantly beyond that period of time.

### Polynomial fitting to waypoints
A 3rd order polynomial is fit to the waypoints received from the simulator (line 113 in `main.cpp`). This polynomial fit is used to calculate the cross-track-error and orientation-error at each of the timesteps in the MPC prediction horizon.

### Latency
In `main.cpp`, line 185, a 100 ms sleep statement is included to simulate a latency between the actuator commands and the actual response of the car to these commands.

Unlike PID control, MPC can directly account for this latency in the kinematic model used for state prediction, thereby compensating for the latency in advance.

This can be achieved by considering a state predicted 100 ms in the future in the calculation of the model constraints for optimization (implemented in lines 134--138 of `MPC.cpp`).

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.

* **Ipopt and CppAD:** Please refer to [this document](https://github.com/udacity/CarND-MPC-Project/blob/master/install_Ipopt_CppAD.md) for installation instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.)
4.  Tips for setting up your environment are available [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)
5. **VM Latency:** Some students have reported differences in behavior using VM's ostensibly a result of latency.  Please let us know if issues arise as a result of a VM environment.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
