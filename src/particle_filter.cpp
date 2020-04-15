/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  default_random_engine gen;
  num_particles = 50;  // Set the number of particles
  normal_distribution<double> dist_x(x,std[0]);
  normal_distribution<double> dist_y(y,std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (unsigned int i = 0;i < num_particles; i++) { 
    
    Particle particle;
    particle.weight = 1.0;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.id = i;
    particles.push_back(particle);
    weights.push_back(particle.weight);
       
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  for (unsigned int i = 0; i < particles.size();i++){
    
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    
    double x_prediction;
    double y_prediction;
    double theta_prediction;
    
    // add check for yaw_rate near zero 
    if (fabs(yaw_rate)<0.0001) { 
      x_prediction =  x + (velocity)*(sin(theta))*delta_t; 
      y_prediction = y + (velocity)*(cos(theta))*delta_t;
      theta_prediction = theta;    
    }
    else {
      x_prediction =  x + (velocity/yaw_rate)*(sin(theta+yaw_rate*delta_t)-sin(theta)); 
      y_prediction = y + (velocity/yaw_rate)*(cos(theta) - cos(theta+yaw_rate*delta_t)); 
      theta_prediction = theta + yaw_rate*delta_t;
    }
        
    //simulate prediction noise
    normal_distribution<double> dist_x(x_prediction,std_pos[0]);
    normal_distribution<double> dist_y(y_prediction,std_pos[1]);
    normal_distribution<double> dist_theta(theta_prediction,std_pos[2]);
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(unsigned int i =0; i<observations.size(); i++) { 
    double observation_x = observations[i].x; 
    double observation_y = observations[i].y;
    int closest_landmark_id;
    double min_distance = 10000000.0;
    for(unsigned int j = 0;j< predicted.size();j++){
      double predicted_x = predicted[j].x;
      double predicted_y = predicted[j].y;
      int landmark_id = predicted[j].id;
      
      double distance = dist(observation_x, observation_y, predicted_x, predicted_y);
      if (distance < min_distance) {
        min_distance = distance;
        closest_landmark_id = landmark_id;
      }
      
    }
    observations[i].id = closest_landmark_id;
    
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   The particles are located according to the MAP'S coordinate system. 
   *   Transofrmation between the two systems is needed (rotation AND translation but not scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double weight_normalizer = 0.0;
  for( unsigned int i =0;i< num_particles; i++) {
    double x_particle = particles[i].x;
    double y_particle = particles[i].y;
    double theta = particles[i].theta;
    
    // Transform observations from vehichle coordinate system to map coordinate system
    vector<LandmarkObs> transformed_observations;
    for(unsigned int j =0;j< observations.size();j++) {
      double x_observation = observations[j].x;
      double y_observation = observations[j].y;
      double x_map = x_particle + cos(theta)*x_observation - sin(theta)*y_observation;
  	  double y_map = y_particle + sin(theta)*x_observation + cos(theta)*y_observation;
      LandmarkObs transformed_observation;
      transformed_observation.x = x_map;
      transformed_observation.y = y_map;
      transformed_observations.push_back(transformed_observation);
      }
    
    // Identify landmarks within sensor range
    vector<LandmarkObs> predicted_landmarks; 
    for(unsigned int j = 0; j< map_landmarks.landmark_list.size();j++) {
      Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
      
      if( (dist(x_particle, y_particle, landmark.x_f,landmark.y_f))<= sensor_range){
        predicted_landmarks.push_back(LandmarkObs {landmark.id_i, landmark.x_f, landmark.y_f});      
      }     
    } 
    // Creates association between landmarks and observations
    dataAssociation(predicted_landmarks, transformed_observations);
    
    // Calculate the weight of each particle (Multivariate Gaussian distribution)
    particles[i].weight = 1.0;
    double sigma_xx = std_landmark[0];
    double sigma_yy = std_landmark[1];
    double two_times_sigma_xx_2 = 2.0*pow(sigma_xx, 2);
    double two_times_sigma_yy_2 = 2.0*pow(sigma_yy, 2);
    double normalizing_term = (1.0/(2.0 * M_PI * sigma_xx * sigma_yy));
    for(unsigned int k = 0;k< transformed_observations.size();k++) {
      int associated_id = transformed_observations[k].id;
      double x = transformed_observations[k].x;
      double y = transformed_observations[k].y; 
 
      for(unsigned int l = 0;l<predicted_landmarks.size();l++) {
        if(predicted_landmarks[l].id == associated_id) {
        	double mu_x = predicted_landmarks[l].x;
      		double mu_y = predicted_landmarks[l].y;
    
      		particles[i].weight *= ( normalizing_term * exp( -( ( pow((x-mu_x),2)/(two_times_sigma_xx_2) ) + ( pow((y-mu_y),2)/(two_times_sigma_yy_2) ) ) ) ) ;
          
        }
      }
    
  } 
    weight_normalizer = weight_normalizer + particles[i].weight;
}
  //normalize weights
  for(unsigned int i;i < particles.size();i++) {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
  
                                                                                                                       }

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    std::default_random_engine gen;

    // Create the distribution with those weights
    std::discrete_distribution<> d(weights.begin(), weights.end());

	int particle_index; 
  	vector<Particle> new_particles;
    Particle particle;
    double normalizer;
    for(int n=0; n<num_particles; n++) {
      particle_index = d(gen);
      particle = particles[particle_index];
      new_particles.push_back(particle);
      normalizer+=particle.weight;
    }
  
  particles= new_particles; 
  for( unsigned int i =0; i<num_particles ;i++){
    	particles[i].weight /=normalizer;
    	weights[i] = particles[i].weight;
    }
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}