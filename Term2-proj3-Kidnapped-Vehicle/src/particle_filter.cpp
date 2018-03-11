/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.

  num_particles = 100;

  default_random_engine gen;
  normal_distribution<double> N_x_init(0, std[0]);
  normal_distribution<double> N_y_init(0, std[1]);
  normal_distribution<double> N_theta_init(0, std[2]);

  for (size_t i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = x + N_x_init(gen);
    p.y = y + N_y_init(gen),
    p.theta = theta + N_theta_init(gen);
    p.weight = 1.0;

    particles.emplace_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

  default_random_engine gen;
  normal_distribution<double> N_x_init(0, std_pos[0]);
  normal_distribution<double> N_y_init(0, std_pos[1]);
  normal_distribution<double> N_theta_init(0, std_pos[2]);

  if(abs(yaw_rate) < 1e-5) { // Vehicle not turning
    for (size_t i = 0; i < num_particles; ++i) {
      double theta = particles[i].theta;
      particles[i].x += velocity * delta_t * cos(theta) + N_x_init(gen);
      particles[i].y += velocity * delta_t * sin(theta) + N_y_init(gen);
      particles[i].theta = theta + N_theta_init(gen);
    }

  } else {

    for (size_t i = 0; i < num_particles; ++i) {
      auto theta = particles[i].theta;
      auto theta_pred = theta + delta_t*yaw_rate;

      particles[i].x += velocity/yaw_rate * (sin(theta_pred) - sin(theta)) +
                        N_x_init(gen);

      particles[i].y += velocity/yaw_rate * (cos(theta) - cos(theta_pred)) +
                        N_y_init(gen);

      particles[i].theta = theta_pred + N_theta_init(gen);
    }
  }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted,
                                     std::vector<LandmarkObs>& observations_map) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the id of the landmark to the observation

  for(auto& observation : observations_map)
  {
    double distance = std::numeric_limits<float>::max();
    for(const auto& pred : predicted)
    {
      double dist_obs_pred = dist(observation.x, observation.y, pred.x, pred.y);
      if(dist_obs_pred < distance)
      {
        observation.id = pred.id;
        distance = dist_obs_pred;
      }
    }
  }
}

void ParticleFilter::transformObsToMapFrame(const Particle& particle,
                                         const std::vector<LandmarkObs>& observations,
                                         std::vector<LandmarkObs>& observations_map)
{
  auto x_p = particle.x;
  auto y_p = particle.y;
  auto theta = particle.theta;

  observations_map.clear();
  LandmarkObs lobs;
  for(const auto& observation : observations) {

    lobs.id = -1;
    auto x = observation.x;
    auto y = observation.y;

    lobs.x = x_p + x*cos(theta) + y*-sin(theta);
    lobs.y = y_p + x*sin(theta) + y*cos(theta);
    observations_map.push_back(lobs);
  }
}

double ParticleFilter::calcParticleWeight(std::vector<LandmarkObs> predicted_obs,
                                          std::vector<LandmarkObs> observations_map,
                                          double std_landmark[])
{
  double weight = 1.0;
  double mu_x, mu_y;

  auto sig_x = std_landmark[0];
  auto sig_y = std_landmark[1];

  for(auto& observation : observations_map)
  {
    auto x_obs = observation.x;
    auto y_obs = observation.y;

    for(const auto& pred : predicted_obs)
    {
      if(pred.id==observation.id)
      {
        mu_x = pred.x;
        mu_y = pred.y;
        break;
      }
    }

    // calculate normalization term
    double gauss_norm = 1/(2 * M_PI * sig_x * sig_y);

    // calculate exponent
    double exponent = pow((x_obs - mu_x), 2)/(2 * pow(sig_x, 2)) +
                      pow((y_obs - mu_y), 2)/(2 * pow(sig_y, 2));

    // calculate weight using normalization terms and exponent
    weight *= gauss_norm * exp(-exponent);
  }

  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  weights.clear();
  for (auto& particle : particles) {

    std::vector<LandmarkObs> observations_map;
    std::vector<LandmarkObs> predicted_obs;

    // Transform observations from Sensor-frame to Map-frame
    transformObsToMapFrame(particle, observations, observations_map);

    // Predicted observations (must be in Map-frame too!)
    LandmarkObs lobs;
    for (const auto& landmark : map_landmarks.landmark_list) {

      if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y)
          <= sensor_range) {
        lobs.id = landmark.id_i;
        lobs.x = landmark.x_f;
        lobs.y = landmark.y_f;

        predicted_obs.push_back(lobs);
      }
    }

    dataAssociation(predicted_obs, observations_map);
    SetAssociations(particle, observations_map);

    // Calculate particle weights
    weights.push_back(calcParticleWeight(predicted_obs, observations_map,
                                         std_landmark));
  }
}

void ParticleFilter::resample() {

  default_random_engine gen;
  std::discrete_distribution<int> distr (weights.begin(), weights.end());

  std::vector<Particle> particles_resampled;
  for(size_t i = 0; i < particles.size(); ++i) {
    particles_resampled.push_back(particles[distr(gen)]);
  }

  particles = particles_resampled;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const std::vector<LandmarkObs>& lobs) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  for(const auto& obs : lobs) {
    particle.associations.push_back(obs.id);
    particle.sense_x.push_back(obs.x);
    particle.sense_y.push_back(obs.y);
  }
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
