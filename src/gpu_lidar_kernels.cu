#include <cstdint>
#include "gpu_lidar_2d.hpp"
__global__ void traceRays(uint8_t *map, float *intersections, int width, int height, int x, int y, float theta, int raycount, float maxDistance)
{
    int rayId = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayId >= raycount)
        return;
    float ray_theta = (float)rayId * 2.0f * 3.14159265359f / (float)raycount;
    float dx = cosf(theta + ray_theta);
    float dy = sinf(theta + ray_theta);
    float current_x = (float)x + (10 * dx);
    float current_y = (float)y + (10 * dy);
    while (true)
    {
        float d = (current_x - (float)x) * (current_x - (float)x) + (current_y - (float)y) * (current_y - (float)y);
        if (d > maxDistance * maxDistance)
        {
            break;
        }
        int map_x = (int)current_x;
        int map_y = (int)current_y;
        if (map_x < 0 || map_x >= width || map_y < 0 || map_y >= height)
        {
            break;
        }
        float pixel_value = (float)map[map_y * width + map_x];
        if (pixel_value < 30.0f)
        {
            intersections[rayId * 2] = map_x;
            intersections[rayId * 2 + 1] = map_y;
            return;
        }
        current_x += dx;
        current_y += dy;
    }
    intersections[rayId * 2] = 0.0f;
    intersections[rayId * 2 + 1] = 0.0f;
}

void GpuLidar2D::simulateLidar(int x, int y, float theta, float max_distance)
{
    int n_blocks = (ray_count + 63) / 64;
    traceRays<<<n_blocks, 64>>>(map_data, d_intersections, map_width, map_height, x, y, theta, ray_count, max_distance);
    cudaMemcpy(h_intersections, d_intersections, 2 * ray_count * sizeof(float), cudaMemcpyDeviceToHost);
}