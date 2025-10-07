#include "stb_image.h"
#include "gpu_lidar_2d.hpp"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "robot.hpp"

GpuLidar2D *lidarPtr;
RobotState goalState;

bool ready = false;
static void onMouse(int event, int x, int y, int, void *)
{
    // lidarPtr->simulateLidar(x, y, 0.0f, 300.0f);
    goalState.x = static_cast<float>(x);
    goalState.y = static_cast<float>(y);
    ready = true;
}

int main()
{
    int width, height, channels;
    unsigned char *data = stbi_load("data/map.png", &width, &height, &channels, 0);
    if (!data)
    {
        std::cout << "Failed to load image" << std::endl;
        throw std::runtime_error("Failed to load image");
    }
    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
    cv::Mat img(height, width, (channels == 1) ? CV_8UC1 : CV_8UC3, data);
    if (channels == 1)
    {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    Robot diffBot(data, width, height, 177.0f, 229.0f, 0.0f);
    int ray_count;
    int key = 0;
    cv::imshow("Lidar Simulation", img);
    cv::setMouseCallback("Lidar Simulation",onMouse);
    while (true)
    {
        cv::Mat img_copy = img.clone();
        if (ready == true)
        {
            diffBot.setHeading(goalState.x, goalState.y);
            diffBot.updatePosition(0.16f);
            diffBot.renderBot(img_copy);
            float *distances = diffBot.getLidarData(ray_count);
            for (int i = 0; i < ray_count; ++i)
            {
                cv::circle(img_copy, cv::Point((int)distances[i * 2], (int)distances[i * 2 + 1]), 1, cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::imshow("Lidar Simulation", img_copy);
        int key = cv::waitKey(16);
        if (key == 27) // ESC key
            break;
    }
    stbi_image_free(data);
    return 0;
}