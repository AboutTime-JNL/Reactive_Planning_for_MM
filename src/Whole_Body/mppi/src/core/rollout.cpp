#include "core/rollout.h"

using namespace mppi;

Rollout::Rollout() {}

Rollout::Rollout(unsigned int steps, unsigned int input_dim,
                 unsigned int state_dim) {
    steps_ = steps;
    input_dim_ = input_dim;
    state_dim_ = state_dim;
    cc = Eigen::VectorXd::Zero(steps);
    tt.resize(steps, 0.0);
    uu.resize(steps, Eigen::VectorXd::Zero(input_dim));
    nn.resize(steps, Eigen::VectorXd::Zero(input_dim));
    xx.resize(steps_, Eigen::VectorXd::Zero(state_dim));
}

void Rollout::clear() {
    total_cost = 0.0;
    tt.resize(steps_, 0.0);
    cc.setZero();
    uu.clear();
    uu.resize(steps_, Eigen::VectorXd::Zero(input_dim_));
    nn.clear();
    nn.resize(steps_, Eigen::VectorXd::Zero(input_dim_));
    xx.clear();
    xx.resize(steps_, Eigen::VectorXd::Zero(state_dim_));
}

void Rollout::clear_cost() {
    total_cost = 0.0;
    cc.setZero();
}

bool Rollout::operator<(const Rollout& roll) const {
    return (total_cost < roll.total_cost);
}

std::ostream& operator<<(std::ostream& os, const mppi::Rollout& r) {
    os << "[t]: ";
    for (const auto& t : r.tt) os << t << ", ";
    os << std::endl;

    os << "[x]: ";
    for (const auto& x : r.xx) os << "(" << x.transpose() << ") ";
    os << std::endl;

    os << "[u]: ";
    for (const auto& u : r.uu) os << "(" << u.transpose() << ") ";
    os << std::endl;

    os << "[e]: ";
    for (const auto& e : r.nn) os << "(" << e.transpose() << ") ";
    os << std::endl;
    os << "Total cost: " << r.total_cost << std::endl;
}