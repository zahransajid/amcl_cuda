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
      state_{start_x, start_y, start_theta}, goal_{start_x, start_y} {}

Robot::~Robot() {}

void Robot::setHeading(float goal_x, float goal_y) {
  goal_.x = goal_x;
  goal_.y = goal_y;
}

void Robot::updatePosition(float dt) {
  float distance_to_goal = std::hypot(goal_.x - state_.x, goal_.y - state_.y);
  if (distance_to_goal < 5.0f)
    return;
  float desired_theta = atan2(goal_.y - state_.y, goal_.x - state_.x);
  float theta_error = normalizeAngle(desired_theta - (state_.theta));
  float desired_angular_vel =
      (theta_error / M_PI) * (max_motor_velocity / (wheel_base_width / 2.0f));

  float linear_velocity = 0.0f;
  if (std::abs(theta_error) < 0.5 &&
      std::hypot(goal_.x - state_.x, goal_.y - state_.y) > 10.0f) {
    linear_velocity = std::min(max_motor_velocity, distance_to_goal / dt);
  }

  float dphi = desired_angular_vel * dt;
  float dx = linear_velocity * cosf(state_.theta) * dt;
  float dy = linear_velocity * sinf(state_.theta) * dt;

  state_.x += dx;
  state_.y += dy;
  state_.theta += dphi;
  dr_state_.x += dx;
  dr_state_.y += dy;
  dr_state_.theta += dphi;
}

void Robot::renderBot(cv::Mat &image) {
  cv::circle(image, cv::Point2f(state_.x, state_.y), 10, cv::Scalar(0, 255, 0),
             -1);
  cv::Point2f line_end = cv::Point2f(state_.x + 30 * cosf(state_.theta),
                                     state_.y + 30 * sinf(state_.theta));
  cv::line(image, cv::Point2f(state_.x, state_.y), line_end,
           cv::Scalar(0, 255, 0), 2);
}

float *Robot::getLidarData(int &num_points) {
  lidar.simulateLidar(static_cast<int>(state_.x), static_cast<int>(state_.y),
                      state_.theta, 200.0f);
  return lidar.getPoints(num_points);
}

RobotState Robot::getState() { return state_; }

void Robot::renderLidar(cv::Mat &image) {
  int num_points;
  float *distances = this->getLidarData(num_points);
  for (int i = 0; i < num_points; ++i) {
    float x = distances[i * 2];
    float y = distances[i * 2 + 1];
    if (x == 0.0f && y == 0.0f) {
      continue;
    }
    float rx = cosf(-state_.theta) * x + sinf(-state_.theta) * y;
    float ry = -sinf(-state_.theta) * x + cosf(-state_.theta) * y;
    cv::circle(image,
               cv::Point(static_cast<int>(rx + state_.x),
                         static_cast<int>(ry + state_.y)),
               1, cv::Scalar(0, 0, 255), -1);
  }
}

void Robot::renderParticles(cv::Mat &image, Particle *particles,
                            int particle_count, int top) {
  for (int i = 0; i < top && i < particle_count; ++i) {
    const Particle &p = particles[i];
    cv::circle(image, cv::Point2f(p.state.x, p.state.y), 2,
               cv::Scalar(255, 0, 0), -1);
  }
}

RobotState Robot::getDRState() { return dr_state_; }

void Robot::renderEstimatedPose(cv::Mat &image, RobotState estimated_pose) {
  cv::Point2f line_end =
      cv::Point2f(estimated_pose.x + 30 * cosf(estimated_pose.theta),
                  estimated_pose.y + 30 * sinf(estimated_pose.theta));
  cv::line(image, cv::Point2f(estimated_pose.x, estimated_pose.y), line_end,
           cv::Scalar(255, 255, 0), 2);
}