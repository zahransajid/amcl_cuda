#include "gpu_lidar_2d.hpp"
#include <cstdint>
__global__ void traceRaysKernel(uint8_t *map, float *intersections, int width,
                                int height, int x, int y, float theta,
                                int raycount, float maxDistance) {
  int rayId = blockIdx.x * blockDim.x + threadIdx.x;
  if (rayId >= raycount)
    return;
  float ray_theta = (float)rayId * 2.0f * 3.14159265359f / (float)raycount;
  float dx = cosf(theta + ray_theta);
  float dy = sinf(theta + ray_theta);
  float current_x = (float)x + (10 * dx);
  float current_y = (float)y + (10 * dy);
  while (true) {
    float d = (current_x - (float)x) * (current_x - (float)x) +
              (current_y - (float)y) * (current_y - (float)y);
    if (d > maxDistance * maxDistance) {
      break;
    }
    int map_x = (int)current_x;
    int map_y = (int)current_y;
    if (map_x < 0 || map_x >= width || map_y < 0 || map_y >= height) {
      break;
    }
    float pixel_value = (float)map[map_y * width + map_x];
    if (pixel_value < 30.0f) {
      {
        float dx_map = (float)map_x - (float)x;
        float dy_map = (float)map_y - (float)y;
        float rx = cosf(theta) * dx_map + sinf(theta) * dy_map;
        float ry = -sinf(theta) * dx_map + cosf(theta) * dy_map;
        intersections[rayId * 2] = rx;
        intersections[rayId * 2 + 1] = ry;
        return;
      }
      intersections[rayId * 2 + 1] = map_y - y;
      return;
    }
    current_x += dx;
    current_y += dy;
  }
  intersections[rayId * 2] = 0.0f;
  intersections[rayId * 2 + 1] = 0.0f;
}

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

void GpuLidar2D::simulateLidar(int x, int y, float theta, float max_distance) {
  int n_blocks = (ray_count_ + 63) / 64;
  traceRaysKernel<<<n_blocks, 64>>>(map_data_, d_intersections_, map_width_,
                                    map_height_, x, y, theta, ray_count_,
                                    max_distance);
  cudaMemcpy(h_intersections_, d_intersections_, 2 * ray_count_ * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GpuLidar2D::correlateParticles(Particle *h_particles, int particle_count) {
  if (particle_count != this->particle_count_) {
    this->particle_count_ = particle_count;
    if (d_particle_data_) {
      cudaFree(d_particle_data_);
    }
    cudaMalloc(&d_particle_data_, particle_count * sizeof(Particle));
  }
  cudaMemcpy(d_particle_data_, h_particles, particle_count * sizeof(Particle),
             cudaMemcpyHostToDevice);
  int n_blocks = (particle_count + 63) / 64;
  correlateParticleKernel<<<n_blocks, 64>>>(map_data_, map_width_, map_height_,
                                            particle_count, d_particle_data_,
                                            d_intersections_, ray_count_);
  cudaMemcpy(h_particles, d_particle_data_, particle_count * sizeof(Particle),
             cudaMemcpyDeviceToHost);
}