#pragma once
#include "gpu_lidar_2d.hpp"
#include <opencv2/opencv.hpp>

struct RobotState {
  float x = 0.0f;
  float y = 0.0f;
  float theta = 0.0f;
};

class Robot {
private:
  GpuLidar2D lidar;
  RobotState state;
  RobotState dr_state; // State from dead reckoning
  float goal_x, goal_y;
  // We'll assume a pixel-to-centimeter ratio of 1:10 for simplicity
  float max_motor_rpm = 50.0f;    // in rpm
  float wheel_base_width = 20.0f; // in cm
  float wheel_radius = 3.0f;      // in cm
  float max_motor_velocity =
      (2 * 3.1415926f * wheel_radius * max_motor_rpm) / 60.0f; // in cm/s

public:
  Robot(uint8_t *map_data, int map_width, int map_height, float start_x,
        float start_y, float start_theta);
  void setHeading(float goal_x, float goal_y);
  void updatePosition(float dt);
  void renderBot(cv::Mat image);
  float *getLidarData(int &num_points);
  // RobotState getState();

  ~Robot();
};
