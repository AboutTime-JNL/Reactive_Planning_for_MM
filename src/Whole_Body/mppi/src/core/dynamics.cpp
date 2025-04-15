#include "core/dynamics.h"

namespace mppi {

Dynamics::Dynamics() {
    // init model
    x_ = observation_t::Zero(Dim::STATE_DIMENSION);
    t_ = 0.0;
}

observation_t Dynamics::step(const input_t& u, const double dt) {
    // integrate joint velocities
    x_.tail<6>() += u.tail<6>() * dt;

    // base velocity in in body frame
    double& yaw = x_(2);
    const double& vx = u(0);
    const double& vy = u(1);
    const double& yawd = u(2);

    x_(0) += (vx * std::cos(yaw) - vy * std::sin(yaw)) * dt;
    x_(1) += (vx * std::sin(yaw) + vy * std::cos(yaw)) * dt;
    x_(2) += yawd * dt;

    t_ += dt;

    return x_;
}

void Dynamics::reset(const observation_t& x, const double t) {
    x_ = x;
    t_ = t;
}

}  // namespace mppi