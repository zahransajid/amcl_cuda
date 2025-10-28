#pragma once

#include "pf_config.hpp"
#include "pf_types.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

__global__ void correlateParticleKernel(uint8_t *map, int width, int height,
                                        int particle_count,
                                        Particle *particle_data,
                                        float *lidar_data, int ray_count);

class ParticleFilter {
public:
  ParticleFilter(uint8_t *map_data, int map_width, int map_height);
  ~ParticleFilter();
  Particle *getParticles();
  int getParticleCount();
  void initializeParticles();
  void updatePositions(RobotState movement, float dt);
  void resample();
  void sortParticles();
  void correlateParticles(float *sensor_data, int ray_count);

private:
  pf_config config_;
  std::vector<Particle> particles_;
  int map_width_;
  int map_height_;
  std::vector<uint8_t> map_data_;
  uint8_t *d_map_data_;
  Particle *d_particle_data_;
  int d_particle_count_;
  float *d_lidar_data_;
  int ray_count_ = 0;
  RobotState previous_movement_;
  bool is_first_update_ = true;

  void normalizeWeights();
};