#include "pf.hpp"
#include "cuda_safety.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>

ParticleFilter::ParticleFilter(uint8_t* map_data, int map_width, int map_height) :
    map_width_(map_width), map_height_(map_height)
{
    this->map_data_.reserve(map_width_ * map_height_);
    std::copy(map_data, map_data + map_width_ * map_height_, std::back_inserter(this->map_data_));
    particles_.resize(config_.MAX_PARTICLE_COUNT);
    CUDA_SAFE_CALL(cudaMalloc(&d_map_data_, map_width_ * map_height_ * sizeof(uint8_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_map_data_, map_data_.data(),
                              map_width_ * map_height_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc(&d_particle_data_, config_.MAX_PARTICLE_COUNT * sizeof(Particle)));
}

ParticleFilter::~ParticleFilter()
{
    CUDA_SAFE_CALL(cudaFree(d_map_data_));
    CUDA_SAFE_CALL(cudaFree(d_particle_data_));
    if (ray_count_ > 0)
        CUDA_SAFE_CALL(cudaFree(d_lidar_data_));
}

void ParticleFilter::initializeParticles()
{
    is_first_update_ = true;
    int n = 0;
    while (n < config_.MAX_PARTICLE_COUNT)
    {
        int x = rand() % map_width_;
        int y = rand() % map_height_;
        if (map_data_[y * map_width_ + x] > 200)
        {
            particles_[n].state.x = static_cast<float>(x);
            particles_[n].state.y = static_cast<float>(y);
            particles_[n].state.theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.1415926f;
            particles_[n].state.theta = 0.0f;
            particles_[n].weight = 1.0 / config_.MAX_PARTICLE_COUNT;
            n++;
        }
    }
}

void ParticleFilter::sortParticles()
{
    std::sort(particles_.begin(), particles_.end(),
              [](const Particle& a, const Particle& b) { return a.weight > b.weight; });
}

void ParticleFilter::normalizeWeights()
{
    double total_weight = 0.0;
    for (const auto& particle : particles_)
    {
        total_weight += particle.weight;
    }
    for (auto& particle : particles_)
    {
        particle.weight /= total_weight;
    }
}

void ParticleFilter::resample()
{
    this->normalizeWeights();
    std::vector<double> cumulative_weights;
    cumulative_weights.reserve(particles_.size());
    std::vector<Particle> new_particles;
    new_particles.reserve(particles_.size());
    double running_sum = 0.0;
    for (const auto& particle : particles_)
    {
        running_sum += particle.weight;
        cumulative_weights.push_back(running_sum);
    }

    Particle p;
    float noise_dx, noise_dy, noise_dtheta;
    int actual_resample_count = static_cast<int>(particles_.size() * config_.RESAMPLE_THRESHOLD);
    for (size_t i = 0; i < actual_resample_count; ++i)
    {
        float random_value = static_cast<float>(rand()) / RAND_MAX;
        auto it =
            std::lower_bound(cumulative_weights.begin(), cumulative_weights.end(), random_value);
        size_t index = std::distance(cumulative_weights.begin(), it);
        if (index >= particles_.size())
        {
            index = particles_.size() - 1;
        }
        p = particles_[index];
        new_particles.push_back(p);
    }
    for (size_t i = 0; i < particles_.size() - actual_resample_count; i++)
    {
        float random_value = static_cast<float>(rand()) / RAND_MAX;
        auto it =
            std::lower_bound(cumulative_weights.begin(), cumulative_weights.end(), random_value);
        size_t index = std::distance(cumulative_weights.begin(), it);
        if (index >= particles_.size())
        {
            index = particles_.size() - 1;
        }
        p = particles_[index];
        noise_dx = std::normal_distribution<float>(0.0f, config_.POSITION_NOISE_STD_DEV)(
            std::default_random_engine());
        noise_dy = std::normal_distribution<float>(0.0f, config_.POSITION_NOISE_STD_DEV)(
            std::default_random_engine());
        noise_dtheta = std::normal_distribution<float>(0.0f, config_.ANGLE_NOISE_STD_DEV)(
            std::default_random_engine());
        p.state.x += noise_dx;
        p.state.y += noise_dy;
        p.state.theta += noise_dtheta;
        new_particles.push_back(p);
    }
    particles_ = std::move(new_particles);
}

Particle* ParticleFilter::getParticles()
{
    return particles_.data();
}
int ParticleFilter::getParticleCount()
{
    return static_cast<int>(particles_.size());
}

void ParticleFilter::updatePositions(RobotState movement, float dt)
{
    if (is_first_update_)
    {
        previous_movement_ = movement;
        is_first_update_ = false;
        return;
    }
    float dx = movement.x - previous_movement_.x;
    float dy = movement.y - previous_movement_.y;
    float dtheta = movement.theta - previous_movement_.theta;
    for (auto& particle : particles_)
    {

        particle.state.x += dx;
        particle.state.y += dy;
        particle.state.theta += dtheta;
        previous_movement_ = movement;
    }
}

RobotState ParticleFilter::calculateEstimatedPose()
{
    RobotState estimated_pose;
    int count = static_cast<int>(particles_.size() * config_.CALCULATE_POSE_POPULATION_FRACTION);
    int n = static_cast<int>(particles_.size());
    count = std::max(count, 1);
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_theta = 0.0f;
    // We assume its sorted here
    for (int i = 0; i < count; ++i)
    {
        sum_x += particles_[i].state.x;
        sum_y += particles_[i].state.y;
        sum_theta += particles_[i].state.theta;
    }
    estimated_pose.x = sum_x / count;
    estimated_pose.y = sum_y / count;
    estimated_pose.theta = sum_theta / count;
    return estimated_pose;
}