#include "gpu_lidar_2d.hpp"

GpuLidar2D::GpuLidar2D(uint8_t *data, int width, int height, int num_rays) {
  cudaMalloc(&d_map_data_, width * height * sizeof(uint8_t));
  cudaMemcpy(d_map_data_, data, width * height * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_intersections_, num_rays * sizeof(float) * 2);
  h_intersections_.reserve(num_rays * 2);
  ray_count_ = num_rays;
  map_width_ = width;
  map_height_ = height;
}

GpuLidar2D::~GpuLidar2D() {
  cudaFree(d_map_data_);
  cudaFree(d_intersections_);
}

float *GpuLidar2D::getPoints(int &num_points) {
  num_points = ray_count_;
  return h_intersections_.data();
}
