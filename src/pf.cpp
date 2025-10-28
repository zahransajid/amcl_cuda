#include "pf.hpp"

ParticleFilter::ParticleFilter(uint8_t *map_data, int map_width, int map_height)
    : map_width_(map_width), map_height_(map_height), map_data_(map_data) {
  particles_.resize(config_.PARTICLE_COUNT);
  this->initializeParticles();
}

void ParticleFilter::initializeParticles() {
  is_first_update_ = true;
  int n = 0;
  while (n < config_.PARTICLE_COUNT) {
    int x = rand() % map_width_;
    int y = rand() % map_height_;
    if (map_data_[y * map_width_ + x] > 200) {
      particles_[n].state.x = static_cast<float>(x);
      particles_[n].state.y = static_cast<float>(y);
      particles_[n].state.theta =
          static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.1415926f;
      particles_[n].state.theta = 0.0f;
      particles_[n].weight = 1.0 / config_.PARTICLE_COUNT;
      n++;
    }
  }
}

void ParticleFilter::drawParticles(cv::Mat &image, int top) {
  if (top > 0) {
    std::sort(particles_.begin(), particles_.end(),
              [](const Particle &a, const Particle &b) {
                return a.weight > b.weight;
              });
    for (int i = 0; i < top; ++i) {
      cv::circle(image,
                 cv::Point2f(particles_[i].state.x, particles_[i].state.y), 2,
                 cv::Scalar(255, 0, 0), -1);
    }
  } else {
    for (const auto &particle : particles_) {
      cv::circle(image, cv::Point2f(particle.state.x, particle.state.y), 2,
                 cv::Scalar(255, 0, 0), -1);
    }
  }
}

void ParticleFilter::normalizeWeights() {
  double total_weight = 0.0;
  for (const auto &particle : particles_) {
    total_weight += particle.weight;
  }
  for (auto &particle : particles_) {
    particle.weight /= total_weight;
  }
}

void ParticleFilter::resample() {
  this->normalizeWeights();
  std::vector<double> cumulative_weights;
  cumulative_weights.reserve(particles_.size());
  std::vector<Particle> new_particles;
  new_particles.reserve(particles_.size());
  double running_sum = 0.0;
  for (const auto &particle : particles_) {
    running_sum += particle.weight;
    cumulative_weights.push_back(running_sum);
  }
  Particle p;
  for (size_t i = 0; i < particles_.size(); ++i) {
    float random_value = static_cast<float>(rand()) / RAND_MAX;
    auto it = std::lower_bound(cumulative_weights.begin(),
                               cumulative_weights.end(), random_value);
    size_t index = std::distance(cumulative_weights.begin(), it);
    if (index >= particles_.size()) {
      index = particles_.size() - 1;
    }
    p = particles_[index];
    // This is wrong, just for testing
    new_particles.push_back(p);
  }
  particles_ = std::move(new_particles);
}

Particle *ParticleFilter::getParticles() { return particles_.data(); }
int ParticleFilter::getParticleCount() {
  return static_cast<int>(particles_.size());
}

void ParticleFilter::updatePositions(RobotState movement, float dt) {
  if (is_first_update_) {
    previous_movement_ = movement;
    is_first_update_ = false;
    return;
  }
  float dx = movement.x - previous_movement_.x;
  float dy = movement.y - previous_movement_.y;
  float dtheta = movement.theta - previous_movement_.theta;
  for (auto &particle : particles_) {
    float noise_x = dx * (static_cast<float>(rand()) / RAND_MAX - 0.5f) *
                    config_.POSITION_NOISE;
    float noise_y = dy * (static_cast<float>(rand()) / RAND_MAX - 0.5f) *
                    config_.POSITION_NOISE;
    float noise_theta = dtheta *
                        (static_cast<float>(rand()) / RAND_MAX - 0.5f) *
                        config_.ANGLE_NOISE;

    particle.state.x += dx + noise_x;
    particle.state.y += dy + noise_y;
    particle.state.theta += dtheta + noise_theta;
    previous_movement_ = movement;
  }
}