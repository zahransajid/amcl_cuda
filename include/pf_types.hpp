#pragma once
struct RobotState
{
    float x = 0.0f;
    float y = 0.0f;
    float theta = 0.0f;
};

struct Particle
{
    RobotState state;
    double weight;
};