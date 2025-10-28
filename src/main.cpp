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
static void onMouse(int event, int x, int y, int, void *) {
  goalState.x = static_cast<float>(x);
  goalState.y = static_cast<float>(y);
  ready = true;
}

int main() {
  int width, height, channels;
  unsigned char *data =
      stbi_load("data/map.png", &width, &height, &channels, 0);
  if (!data) {
    std::cout << "Failed to load image" << std::endl;
    throw std::runtime_error("Failed to load image");
  }
  std::cout << "Image loaded: " << width << "x" << height << " with "
            << channels << " channels." << std::endl;
  cv::Mat img(height, width, (channels == 1) ? CV_8UC1 : CV_8UC3, data);
  if (channels == 1) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
  }
  Robot diffBot(data, width, height, 177.0f, 229.0f, 0.0f);
  ParticleFilter pf(data, width, height);
  pf.initializeParticles();
  int ray_count;
  int key = 0;
  cv::imshow("Lidar Simulation", img);
  cv::setMouseCallback("Lidar Simulation", onMouse);
  int n = 0;
  while (true) {
    n++;
    cv::Mat img_copy = img.clone();
    if (ready == true) {
      diffBot.setHeading(goalState.x, goalState.y);
      // diffBot.setHeading(177.0f, 229.0f);
      diffBot.updatePosition(0.16f);
      pf.updatePositions(diffBot.getDRState(), 0.16f);
      float *distances = diffBot.getLidarData(ray_count);
      Particle a, b, c, d;
      a.state = diffBot.getState();
      b.state = diffBot.getState();
      c.state = diffBot.getState();
      b.state.x += 10.0f;
      c.state.y += 10.0f;
      d.state.x = 0.0f;
      d.state.y = 0.0f;
      d.state.theta = 0.0f;
      // Particle particles_array[4] = {a,b,c,d};
      // diffBot.lidar.correlateParticles(particles_array, 4);
      // std::cout << "Particle correlations: " << particles_array[0].weight <<
      // ", "
      //           << particles_array[1].weight << ", "
      //           << particles_array[2].weight << ", "
      //           << particles_array[3].weight << std::endl;
      diffBot.renderBot(img_copy);
      diffBot.renderLidar(img_copy);
      RobotState robot_state;
      robot_state = diffBot.getState();
    }
    diffBot.lidar.correlateParticles(pf.getParticles(), pf.getParticleCount());

    // for(int i = 0; i < 5; ++i)
    // {
    //   std::cout << "Top particle " << i << " weight: " <<
    //   pf.getParticles()[i].weight << std::endl;
    // }
    if (n % 50 == 0) {
      pf.resample();
    }
    pf.drawParticles(img_copy, 100);
    cv::imshow("Lidar Simulation", img_copy);
    int key = cv::waitKey(16);
    if (key == 27) // ESC key
      break;
  }
  stbi_image_free(data);
  return 0;
}