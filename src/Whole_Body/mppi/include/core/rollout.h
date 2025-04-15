#pragma once
#include <Eigen/Core>
#include <vector>

namespace mppi {
/// Structure to collect the rollout information
/// can store one trajectory
struct Rollout {
    Rollout();
    explicit Rollout(unsigned int steps, unsigned int input_dim,
                     unsigned int state_dim);

    bool valid = true;
    unsigned int steps_;
    unsigned int input_dim_;
    unsigned int state_dim_;
    double total_cost = 0.0;

    Eigen::VectorXd cc;
    std::vector<double> tt;
    std::vector<Eigen::VectorXd> uu;
    std::vector<Eigen::VectorXd> nn;
    std::vector<Eigen::VectorXd> xx;

    void clear();
    void clear_cost();

    bool operator<(const Rollout& roll) const;
};
}  // namespace mppi

std::ostream& operator<<(std::ostream& os, const mppi::Rollout& r);