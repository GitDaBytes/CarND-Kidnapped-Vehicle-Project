/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 1000;

	// split out for readability
	double std_dev_x = std[0];
	double std_dev_y = std[1];
	double std_dev_theta = std[2];

	// reset vector memory and resources
	std::vector<Particle>().swap(this->particles);

	this->particles.reserve(num_particles); // reserve memory to make insert faster

	// reset weights memory and resources
	std::vector<double>().swap(this->weights);

	this->weights.reserve(num_particles); // reserve memory to make insert faster

	// set up a norm dist for each parameter. We will be sampling from this to setup 
	// num_particles particles with gaussian noise added to the given gps postion for each particle
	normal_distribution<double> dist_x(x, std_dev_x);
	normal_distribution<double> dist_y(y, std_dev_y);
	normal_distribution<double> dist_theta(theta, std_dev_theta);

	// generate x particles setting the postion of each, adding uniform noise to each one
	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);

		weights.push_back(p.weight); // making a seperate list of weights for easier retreival later
	}

	std::cout << "Init complete." << std::endl;

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// split out std dev of position noise for readability
	double std_dev_x = std_pos[0];
	double std_dev_y = std_pos[1];
	double std_dev_theta = std_pos[2];

	if (yaw_rate == 0.0) // put the if outside the loop to minimize number of tests within loop
	{
		for (int i = 0; i < num_particles; i++)
		{
			// use pythag to calculate x/y transition

			double dist = delta_t * velocity; // distance travelled (hypotenuse)

			// update x pos
			particles[i].x += std::cos(particles[i].theta) * dist;

			// update y pos
			particles[i].y += std::sin(particles[i].theta) * dist;

			// theta does not change - yaw rate is zero in this case

			// add gaussian noise to the position / heading
			normal_distribution<double> dist_x(particles[i].x, std_dev_x);
			normal_distribution<double> dist_y(particles[i].y, std_dev_y);
			normal_distribution<double> dist_theta(particles[i].theta, std_dev_theta);


			// finalize pos and heading with noise added
			particles[i].x = dist_x(gen);
			particles[i].y = dist_y(gen);
			particles[i].theta = dist_theta(gen);
		}
	}
	else
	{
		for (int i = 0; i < num_particles; i++)
		{
			// update x pos
			particles[i].x += (velocity / yaw_rate) * (std::sin(particles[i].theta + yaw_rate * delta_t) - std::sin(particles[i].theta));

			// update y pos
			particles[i].y += (velocity / yaw_rate) * (std::cos(particles[i].theta) - std::cos(particles[i].theta + yaw_rate*delta_t));

			// update theta
			particles[i].theta += yaw_rate * delta_t;


			// add gaussian noise to the position / heading
			normal_distribution<double> dist_x(particles[i].x, std_dev_x);
			normal_distribution<double> dist_y(particles[i].y, std_dev_y);
			normal_distribution<double> dist_theta(particles[i].theta, std_dev_theta);

			// finalize pos and heading with noise added
			particles[i].x = dist_x(gen);
			particles[i].y = dist_y(gen);
			particles[i].theta = dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	for (size_t o_idx = 0; o_idx < observations.size(); o_idx++)
	{
		double smallest_dist = std::numeric_limits<double>::max();
		observations[o_idx].id = -1;


		for (size_t p_idx = 0; p_idx < predicted.size(); p_idx++)
		{
			double dist = measure_dist(observations[o_idx].x, observations[o_idx].y, predicted[p_idx].x, predicted[p_idx].y);

			if (dist < smallest_dist)
			{
				observations[o_idx].id = predicted[p_idx].id;
				smallest_dist = dist;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	double particle_weight_norm = 0.0;

	// loop through each particle...
	for (int i = 0; i < num_particles; i++)
	{
		std::vector<LandmarkObs> predicted_obs;

		// generate a list of all map landmarks within sensor range of our expected particle position
		for (size_t i = 0; i < map_landmarks.landmark_list.size(); i++)
		{
			double dist = measure_dist(map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f, particles[i].x, particles[i].y);
			if (dist <= sensor_range)
			{
				LandmarkObs obs;
				obs.id = map_landmarks.landmark_list[i].id_i;
				obs.x = map_landmarks.landmark_list[i].x_f;
				obs.y = map_landmarks.landmark_list[i].y_f;

				predicted_obs.push_back(obs);
			}
		}

		// make a copy of the observed objects 
		std::vector<LandmarkObs> obs2(observations.begin(), observations.end());

		// transform observations from car coords to map coords (the current particle's coords)
		for (size_t idx_obs = 0; idx_obs < obs2.size(); idx_obs++)
		{
			double map_x, map_y;
			transform_coords_car_to_map(particles[i].x, particles[i].y, particles[i].theta,
										obs2[idx_obs].x, obs2[idx_obs].y, map_x, map_y);

			// update observed obs to map coords
			obs2[idx_obs].x = map_x;
			obs2[idx_obs].y = map_y;
		}

		// now associate closest observation to closest prediction using map id
		dataAssociation(predicted_obs, obs2);

		for (size_t idx_obs = 0; idx_obs < obs2.size(); idx_obs++)
		{
			if (idx_obs == 0)
			{
				particles[i].weight = 1.0;
			}

			double target_x = 0.0;
			double target_y = 0.0;

			// find x,y coordinates of the closest prediction for obs
			for (size_t k = 0; k < predicted_obs.size(); k++) 
			{
				if (predicted_obs[k].id == obs2[idx_obs].id)
				{
					target_x = predicted_obs[k].x;
					target_y = predicted_obs[k].y;
				}
			}

			//calculate the weight

			particles[i].weight *= generate_particle_weight(obs2[idx_obs].x, obs2[idx_obs].y,
														target_x, target_y,
														std_landmark[0], std_landmark[1]);
		}

		// keep running track of final weight sum so we can norm all weights later
		particle_weight_norm += particles[i].weight;
	}
	
	// Normalize weights (so weights for all particles sum to 1)

	// stop divide by zero
	if (particle_weight_norm == 0.0) 
		particle_weight_norm = numeric_limits<double>::epsilon();

	// norm all weights and copy to weights only vec. for use in resample
	for (size_t p_idx = 0; p_idx < particles.size(); p_idx++)
	{
		this->particles[p_idx].weight /= particle_weight_norm;
		this->weights[p_idx] = this->particles[p_idx].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<int> weighted_distribution(this->weights.begin(), this->weights.end());

	std::vector<Particle> resampled_p;

	resampled_p.reserve(this->num_particles); // optomize for insert speed

	for (int i = 0; i < this->num_particles; i++) 
	{
		int newIdx = weighted_distribution(gen);
		resampled_p.push_back(this->particles[newIdx]);
	}

	// make our resampled vector the current vector
	particles.swap(resampled_p);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


double ParticleFilter::generate_particle_weight(double observation_x, double observation_y, // coords of observation
												double mu_x, double mu_y, // coords of nearest landmarks
												double landmark_stddev_x, double landmark_stddev_y
)
{
							   // calculate normalization term
	double gauss_norm = (1 / (2 * M_PI * landmark_stddev_x * landmark_stddev_y));

	// calculate exponent
	double exponent = std::pow(observation_x - mu_x, 2.0) / (2 * std::pow(landmark_stddev_x, 2.0)) + std::pow(observation_y - mu_y, 2.0) / (2 * std::pow(landmark_stddev_y, 2.0));


	// calculate weight using normalization terms and exponent
	double weight = gauss_norm * std::exp(-exponent);

	return weight;
}

void ParticleFilter::transform_coords_car_to_map(double current_x, double current_y, double current_theta, double observed_x, double observed_y, double &result_x, double &result_y)
{
	double theta = current_theta;

								   // transform to map x coordinate
	result_x = current_x + (std::cos(theta) * observed_x) - (std::sin(theta) * observed_y);

	// transform to map y coordinate
	result_y = current_y + (std::sin(theta) * observed_x) + (std::cos(theta) * observed_y);
}

double ParticleFilter::measure_dist(double pt1_x, double pt1_y, double pt2_x, double pt2_y)
{
	return sqrt(std::pow(pt1_x - pt2_x, 2.0) + std::pow(pt1_y - pt2_y, 2.0));
}