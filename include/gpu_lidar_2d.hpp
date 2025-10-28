#pragma once
#include "pf_types.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

__global__ void traceRaysKernel(uint8_t *map, float *intersections, int width,
                                int height, int x, int y, float theta,
                                int raycount, float maxDistance);

class GpuLidar2D {
private:
  uint8_t *d_map_data_;
  int map_width_;
  int map_height_;
  int ray_count_;
  float *d_intersections_;
  std::vector<float> h_intersections_;

public:
  GpuLidar2D(uint8_t *data, int width, int height, int num_rays);
  ~GpuLidar2D();

  void simulateLidar(int x, int y, float theta, float max_distance);
  float *getPoints(int &num_points);
};
