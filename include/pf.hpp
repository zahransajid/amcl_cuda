#pragma once

#include "pf_config.hpp"
#include "pf_types.hpp"
#include "robot.hpp"

class ParticleFilter {
public:
  ParticleFilter(uint8_t *map_data, int map_width, int map_height);
  void drawParticles(cv::Mat &image, int top);
  Particle *getParticles();
  int getParticleCount();
  void initializeParticles();
  void updatePositions(RobotState movement, float dt);
  void resample();
  //    void updateWeights(const float* lidar_data, int num_points);
  //    void resample();
  //    RobotState estimate() const;

private:
  pf_config config_;
  std::vector<Particle> particles_;
  int map_width_;
  int map_height_;
  uint8_t *map_data_;
  RobotState previous_movement_;
  bool is_first_update_ = true;

  void normalizeWeights();
};