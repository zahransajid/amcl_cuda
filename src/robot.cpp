#include "robot.hpp"
#include <cmath>
#include <iostream>

const float M_PI = 3.1415926f;

float normalizeAngle(float angle) {
  angle = std::fmod(angle, 2.0f * M_PI);
  if (angle > M_PI)
    angle -= 2.0f * M_PI;
  return angle;
}

Robot::Robot(uint8_t *map_data, int map_width, int map_height, float start_x,
             float start_y, float start_theta)
    : lidar(map_data, map_width, map_height, 1000),
      state{start_x, start_y, start_theta} {}

Robot::~Robot() {}

void Robot::setHeading(float goal_x, float goal_y) {
  this->goal_x = goal_x;
  this->goal_y = goal_y;
}

void Robot::updatePosition(float dt) {
  float distance_to_goal = std::hypot(goal_x - state.x, goal_y - state.y);
  if (distance_to_goal < 5.0f)
    return;
  float desired_theta = atan2(goal_y - state.y, goal_x - state.x);
  float theta_error = normalizeAngle(desired_theta - (state.theta));
  float desired_angular_vel =
      (theta_error / M_PI) * (max_motor_velocity / (wheel_base_width / 2.0f));

  float linear_velocity = 0.0f;
  if (std::abs(theta_error) < 0.5 &&
      std::hypot(goal_x - state.x, goal_y - state.y) > 10.0f) {
    linear_velocity = std::min(max_motor_velocity, distance_to_goal / dt);
  }

  float dphi = desired_angular_vel * dt;
  float dx = linear_velocity * cosf(state.theta) * dt;
  float dy = linear_velocity * sinf(state.theta) * dt;

  state.x += dx;
  state.y += dy;
  state.theta += dphi;
  dr_state.x += dx;
  dr_state.y += dy;
  dr_state.theta += dphi;
}

void Robot::renderBot(cv::Mat image) {
  cv::circle(image, cv::Point2f(state.x, state.y), 10, cv::Scalar(0, 255, 0),
             -1);
  cv::Point2f line_end = cv::Point2f(state.x + 30 * cosf(state.theta),
                                     state.y + 30 * sinf(state.theta));
  cv::line(image, cv::Point2f(state.x, state.y), line_end,
           cv::Scalar(0, 255, 0), 2);
}

float *Robot::getLidarData(int &num_points) {
  lidar.simulateLidar(static_cast<int>(state.x), static_cast<int>(state.y),
                      state.theta, 200.0f);
  return lidar.getPoints(num_points);
}