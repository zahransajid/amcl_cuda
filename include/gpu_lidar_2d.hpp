#pragma once
#include <cstdint>
#include <cuda_runtime.h>

__global__ void traceRays(uint8_t* map, float *intersections, int width, int height, int x, int y, int raycount, float maxDistance);

class GpuLidar2D
{
private:
    uint8_t *map_data, *ray_texture;
    int map_width;
    int map_height;
    int ray_count;
    float *d_intersections;
    float *h_intersections;
public:
    GpuLidar2D(uint8_t *data, int width, int height, int num_rays);
    ~GpuLidar2D();

    void simulateLidar(int x, int y, float theta, float max_distance);
    float *getPoints(int &num_points);
};
