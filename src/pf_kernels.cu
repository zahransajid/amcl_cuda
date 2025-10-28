__global__ void correlateParticleKernel(uint8_t *map, int width, int height,
                                        int particle_count,
                                        Particle *particle_data,
                                        float *lidar_data, int ray_count) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= particle_count)
    return;
  Particle &p = particle_data[pid];
  float score = 0.0f;
  for (int r = 0; r < ray_count; r++) {
    float lidar_x = lidar_data[r * 2];
    float lidar_y = lidar_data[r * 2 + 1];
    if (lidar_x == 0.0f && lidar_y == 0.0f) {
      continue;
    }
    int map_x = (int)(p.state.x + (lidar_x - p.state.x) * cosf(p.state.theta) -
                      (lidar_y - p.state.y) * sinf(p.state.theta));
    int map_y = (int)(p.state.y + (lidar_x - p.state.x) * sinf(p.state.theta) +
                      (lidar_y - p.state.y) * cosf(p.state.theta));

    if (map_x < 0 || map_x >= width || map_y < 0 || map_y >= height) {
      continue;
    }
    float pixel_value = (float)map[map_y * width + map_x];
    if (pixel_value < 30.0f) {
      score += 1.0f;
    }
  }
  p.weight = score;
}