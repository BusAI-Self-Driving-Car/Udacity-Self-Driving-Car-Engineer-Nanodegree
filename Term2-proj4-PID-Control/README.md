# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

### Proportional control
Proportional control generates a control signal in direct proportion to the cross-track-error (CTE), in such a way that the CTE is reduced. For example, if the car is diverging rightwards from the center of the road, P-control will generate a steering angle command in the counter-clockwise direction with the aim of bringing the car back towards the center of the road.

Using a P-control gain `K_p = 1.0` leads to the car rapidly oscillating about the center of the road. At some point, the car drives off the road completely:

#### P-control with parameter 1.0
<p align="center">
 <a href=""><img src="./videos/P-control-1_0.gif" alt="Overview" width="50%" height="50%"></a>
</p>

I gradually reduced the P-control gain down to 0.1 such that the car manages to stay on the road much longer than with `K_p = 1.0`. However P-control by itself is not sufficient, especially as the car starts turning. As the car turns, the reference trajectory changes continuosly, increasing the CTE, and thus the P-control command. This leads to oscillations. To prevent this behavior, I will include Derivative-control which damps the P-control command as
the car gets closer to the reference trajectory (center of the road).

#### P-control with parameter 0.1
<p align="center">
 <a href=""><img src="./videos/P-control-0_1.gif" alt="Overview" width="50%" height="50%"></a>
</p>


### Proportional-derivative control

Derivative(D)-control damps the P-control command as the car gets closer to the center of the road. Thus the car does not overshoot much (ideally not at all). The system is more stable and settles to its desired state quicker than with just P-control. With the following PD-control parameters, it was possible to keep the car on the drivable surface of the road throughout the simulator track.

#### P-control with parameter 0.1 and D-control with parameter 1.0
<p align="center">
 <a href=""><img src="./videos/P-0_1-D-1_0.gif" alt="Overview" width="50%" height="50%"></a>
</p>

### Integral control

Integral(I)-control generates a command component that compensates for any systematic bias in the system. The car could have such a bias if, for example, the front wheels are not oriented exactly straight for a steering angle of 0°. Either the car in the Udacity simulator does not seems to have such a bias, or the simulator track is not long enough for such a bias to have any visible effect. Hence I did not use an I-controller in my project.

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
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`.

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

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
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/e8235395-22dd-4b87-88e0-d108c5e5bbf4/concepts/6a4d8d42-6a04-4aa6-b284-1697c0fd6562)
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

