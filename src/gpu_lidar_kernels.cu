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

void GpuLidar2D::simulateLidar(int x, int y, float theta, float max_distance) {
  int n_blocks = (ray_count_ + 63) / 64;
  traceRaysKernel<<<n_blocks, 64>>>(d_map_data_, d_intersections_, map_width_,
                                    map_height_, x, y, theta, ray_count_,
                                    max_distance);
  cudaMemcpy(h_intersections_.data(), d_intersections_,
             2 * ray_count_ * sizeof(float), cudaMemcpyDeviceToHost);
}