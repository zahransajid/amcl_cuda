#pragma once
#include "pf_types.hpp"
#include <cstdint>
#include <cuda_runtime.h>

__global__ void traceRaysKernel(uint8_t *map, float *intersections, int width,
                                int height, int x, int y, float theta,
                                int raycount, float maxDistance);
__global__ void correlateParticleKernel(uint8_t *map, int width, int height,
                                        int particle_count,
                                        Particle *particle_data,
                                        float *correlation_scores,
                                        uint8_t *lidar_data, int ray_count);

class GpuLidar2D {
private:
  uint8_t *map_data_;
  int map_width_;
  int map_height_;
  int ray_count_;
  float *d_intersections_;
  float *h_intersections_;
  Particle *d_particle_data_;
  int particle_count_ = 0;

public:
  GpuLidar2D(uint8_t *data, int width, int height, int num_rays);
  ~GpuLidar2D();

  void simulateLidar(int x, int y, float theta, float max_distance);
  void correlateParticles(Particle *h_particles, int particle_count);
  float *getPoints(int &num_points);
};
