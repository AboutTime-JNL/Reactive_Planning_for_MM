#pragma once

#include "core/typedefs.h"

namespace mppi {

enum Dim {
    STATE_DIMENSION = 9,     // x, y, yaw, q
    INPUT_DIMENSION = 9,     // x_dot, y_dot, yaw_dot, q_dot
    REFERENCE_DIMENSION = 7  // ee_pos, ee_quat
};

/**
 * @brief         relationship between state and input;
 *                store the state and time;
 *                be used for the simulation and controller interval (dt can be
 * different!!)
 * @attention
 */
class Dynamics {
   public:
    Dynamics();
    ~Dynamics() = default;

    dynamics_ptr create() { return std::make_shared<Dynamics>(); }
    dynamics_ptr clone() const { return std::make_shared<Dynamics>(*this); }

    unsigned int get_state_dimension() { return Dim::STATE_DIMENSION; }
    observation_t get_state() { return x_; }
    double get_time() { return t_; }
    unsigned int get_input_dimension() { return Dim::INPUT_DIMENSION; }
    input_t get_zero_input() { return input_t::Zero(get_input_dimension()); }

    void reset(const observation_t& x, const double t);

    observation_t step(const input_t& u, const double dt);

   private:
    observation_t x_;
    double t_;
};
}  // namespace mppi