#pragma once
#include "gpu_lidar_2d.hpp"
#include "pf_types.hpp"
#include <opencv2/opencv.hpp>

class Robot {
private:
  RobotState state_;
  RobotState dr_state_; // State from dead reckoning
  RobotState goal_;
  // We'll assume a pixel-to-centimeter ratio of 1:10 for simplicity
  float max_motor_rpm = 500.0f;   // in rpm
  float wheel_base_width = 20.0f; // in cm
  float wheel_radius = 3.0f;      // in cm
  float max_motor_velocity =
      (2 * 3.1415926f * wheel_radius * max_motor_rpm) / 60.0f; // in cm/s

public:
  Robot(uint8_t *map_data, int map_width, int map_height, float start_x,
        float start_y, float start_theta);
  GpuLidar2D lidar;
  void setHeading(float goal_x, float goal_y);
  void updatePosition(float dt);
  void renderBot(cv::Mat &image);
  float *getLidarData(int &num_points);
  void renderLidar(cv::Mat &image);
  void renderParticles(cv::Mat &image, Particle *particles, int particle_count,
                       int top);
  void renderEstimatedPose(cv::Mat &image, RobotState estimated_pose);
  RobotState getState();
  RobotState getDRState();
  // RobotState getState();

  ~Robot();
};
