#include "gpu_lidar_2d.hpp"
#include "pf.hpp"
#include "robot.hpp"
#include "stb_image.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

RobotState goalState;

bool ready = false;
static void onMouse(int event, int x, int y, int, void*)
{
    goalState.x = static_cast<float>(x);
    goalState.y = static_cast<float>(y);
    ready = true;
}

int main()
{
    int width, height, channels;
    unsigned char* data = stbi_load("data/map.png", &width, &height, &channels, 0);
    if (!data)
    {
        std::cout << "Failed to load image" << std::endl;
        throw std::runtime_error("Failed to load image");
    }
    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels."
              << std::endl;
    cv::Mat img(height, width, (channels == 1) ? CV_8UC1 : CV_8UC3, data);
    if (channels == 1)
    {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    Robot diffBot(data, width, height, 177.0f, 229.0f, 0.0f);
    ParticleFilter pf(data, width, height);
    pf.initializeParticles();
    int ray_count;
    int key = 0;
    float iter_time;
    cv::imshow("Lidar Simulation", img);
    cv::setMouseCallback("Lidar Simulation", onMouse);
    std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();
    int n = 0;
    while (true)
    {
        n++;
        cv::Mat img_copy = img.clone();
        if (ready == true)
        {
            diffBot.setHeading(goalState.x, goalState.y);
            iter_time = std::chrono::duration_cast<std::chrono::duration<float>>(
                            std::chrono::steady_clock::now() - last_time)
                            .count();
            diffBot.updatePosition(iter_time);
            pf.updatePositions(diffBot.getDRState(), iter_time);
            float* distances = diffBot.getLidarData(ray_count);
            if (n == 0)
                pf.correlateParticles(distances, ray_count);
            diffBot.renderBot(img_copy);
            diffBot.renderLidar(img_copy);

            if (n % 10 == 0)
            {
                pf.resample();
                pf.correlateParticles(distances, ray_count);
            }
            pf.sortParticles();
            RobotState estimated_pose = pf.calculateEstimatedPose();
            diffBot.renderEstimatedPose(img_copy, estimated_pose);
            diffBot.renderParticles(img_copy, pf.getParticles(), pf.getParticleCount(), 100);
        }
        iter_time = std::chrono::duration_cast<std::chrono::duration<float>>(
                        std::chrono::steady_clock::now() - last_time)
                        .count();
        float fps = 1.0f / iter_time;
        last_time = std::chrono::steady_clock::now();
        cv::putText(img_copy, "FPS: " + std::to_string(static_cast<int>(fps)), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Lidar Simulation", img_copy);
        int frame_wait = 16 - static_cast<int>(iter_time * 1000.0f);
        if (frame_wait < 1)
            frame_wait = 1;
        int key = cv::waitKey(frame_wait);
        if (key == 27) // ESC key
            break;
    }
    stbi_image_free(data);
    return 0;
}