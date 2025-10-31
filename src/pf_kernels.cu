#include "pf.hpp"

__global__ void correlateParticleKernel(uint8_t *map, int width, int height,
                                        int particle_count,
                                        Particle *particle_data,
                                        float *lidar_data, int ray_count) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= particle_count)
    return;
  Particle &p = particle_data[pid];
  float score = 0.0f;
  float ct = cosf(-p.state.theta);
  float st = sinf(-p.state.theta);
  for (int r = 0; r < ray_count; r++) {
    float lidar_x = lidar_data[r * 2];
    float lidar_y = lidar_data[r * 2 + 1];
    if (lidar_x == 0.0f && lidar_y == 0.0f) {
      continue;
    }
    int lidar_map_x = (int)(p.state.x + ct * lidar_x + st * lidar_y);
    int lidar_map_y = (int)(p.state.y + (-st) * lidar_x + ct * lidar_y);
    // if (lidar_map_x > 0 && lidar_map_x < width && lidar_map_y > 0 &&
    //     lidar_map_y < height) {
    //   uint8_t pixel_value = map[lidar_map_y * width + lidar_map_x];
    //   if (pixel_value < 30) {
    //     score += 1.0f;
    //   }
    // }

    for (int i = 0; i < 9; i++) {
      int offset_x = (i % 3) - 1;
      int offset_y = (i / 3) - 1;
      int neighbor_x = lidar_map_x + offset_x;
      int neighbor_y = lidar_map_y + offset_y;
      if (neighbor_x < 0 || neighbor_x >= width || neighbor_y < 0 ||
          neighbor_y >= height) {
        continue;
      }
      float pixel_value = (float)map[neighbor_y * width + neighbor_x];
      if (pixel_value < 30.0f) {
        score += 1.0f;
      }
    }
  }
  p.weight = score;
}

void ParticleFilter::correlateParticles(float *sensor_data, int ray_count) {
  if (this->ray_count_ != ray_count) {
    this->ray_count_ = ray_count;
    if (d_lidar_data_ != nullptr)
      cudaFree(d_lidar_data_);
    cudaMalloc(&d_lidar_data_, ray_count_ * 2 * sizeof(float));
  }
  cudaMemcpy(d_particle_data_, this->particles_.data(),
             config_.MAX_PARTICLE_COUNT * sizeof(Particle),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_lidar_data_, sensor_data, ray_count_ * 2 * sizeof(float),
             cudaMemcpyHostToDevice);
  int n_blocks = (config_.MAX_PARTICLE_COUNT + 63) / 64;
  correlateParticleKernel<<<n_blocks, 64>>>(
      d_map_data_, map_width_, map_height_, config_.MAX_PARTICLE_COUNT,
      d_particle_data_, d_lidar_data_, ray_count_);
  cudaMemcpy(this->particles_.data(), d_particle_data_,
             config_.MAX_PARTICLE_COUNT * sizeof(Particle),
             cudaMemcpyDeviceToHost);
}