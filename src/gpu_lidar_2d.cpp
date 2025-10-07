#include "gpu_lidar_2d.hpp"

GpuLidar2D::GpuLidar2D(uint8_t *data, int width, int height, int num_rays) {
  cudaMalloc(&map_data, width * height * sizeof(uint8_t));
  cudaMemcpy(map_data, data, width * height * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_intersections, num_rays * sizeof(float) * 2);
  h_intersections = new float[num_rays * 2];
  ray_count = num_rays;
  map_width = width;
  map_height = height;
}

GpuLidar2D::~GpuLidar2D() {
  cudaFree(map_data);
  cudaFree(d_intersections);
}

float *GpuLidar2D::getPoints(int &num_points) {
  num_points = ray_count;
  return h_intersections;
}