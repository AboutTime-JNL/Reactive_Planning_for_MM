#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "core/config.h"
#include "core/cost.h"
#include "core/dynamics.h"
#include "core/rollout.h"
#include "core/solver.h"

#include "utils/savgol_filter.h"

namespace mppi {

Solver::Solver(dynamics_ptr dynamics, cost_ptr cost, policy_ptr policy,
               const Config& config)
    : dynamics_(std::move(dynamics)),
      cost_(std::move(cost)),
      policy_(std::move(policy)),
      config_(config) {
    init_data();
    init_threading();
}

void Solver::init_data() {
    // TODO(giuseppe) this should be automatically computed when config is
    // parsed
    steps_ = static_cast<int>(std::ceil(config_.horizon / config_.step_size));
    nx_ = dynamics_->get_state_dimension();
    nu_ = dynamics_->get_input_dimension();

    // best trajectory
    opt_roll_ = Rollout(steps_, nu_, nx_);
    opt_roll_cache_ = Rollout(steps_, nu_, nx_);

    // all trajectories
    weights_.resize(config_.rollouts, 1.0 / config_.rollouts);
    rollouts_.resize(config_.rollouts, Rollout(steps_, nu_, nx_));
    cached_rollouts_ = std::ceil(config_.caching_factor * config_.rollouts);

    observation_set_ = false;
    reference_set_ = false;
}

void Solver::init_threading() {
    if (config_.threads > 1) {
        pool_ = std::make_unique<ThreadPool>(config_.threads);
        futures_.resize(config_.threads);

        for (unsigned int i = 0; i < config_.threads; i++) {
            dynamics_v_.push_back(dynamics_->create());
            cost_v_.push_back(cost_->create());
            cost_v_[i]->esdf_map_ = cost_->esdf_map_;
        }
    }
}

void Solver::update_policy() {
    if (!reference_set_) {
        ROS_ERROR_STREAM("Reference has never been set. Dropping update");
    } else {
        // 外界观测更新，需要对应时间和状态（外界观测更新频率和预测时域间隔可能不一致）
        copy_observation();

        // cost_->publish_sphere(x0_internal_);
        for (unsigned int i = 0; i < config_.substeps; i++) {
            prepare_rollouts();
            // change the base aim
            if (local_minima_mode && !base_path.empty()) {
                Eigen::Vector2d pos;
                pos << x0_internal_[0], x0_internal_[1];

                Eigen::Vector2d error = pos - base_path.back();
                Eigen::Vector2d last_error = last_base_point - base_path.back();

                Eigen::Vector2d aim_dir = base_path.back() - last_base_point;
                aim_dir = aim_dir.normalized();

                cost_->base_aim = base_path.back() + 1.0 * aim_dir;

                if (!cost_->local_track_mode) {
                    cost_->local_track_mode = true;
                    std::cout << "enter local track mode" << std::endl;
                }

                if (error.dot(last_error) < 0 && base_path.size() > 1) {
                    last_base_point = base_path.back();
                    base_path.pop_back();
                }
            } else {
                if (cost_->local_track_mode) {
                    cost_->local_track_mode = false;
                    std::cout << "out local track mode" << std::endl;
                }
            }
            update_reference();
            sample_trajectories();
            optimize();
            filter_input();

            cost_->last_u = opt_roll_.uu[0];
        }
        swap_policies();
    }
}

void Solver::set_observation(const observation_t& x, const double t) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    x0_ = x;
    reset_time_ = t;
    lock.unlock();

    // initialization of rollouts data
    if (first_step_) {
        copy_observation();
        initialize_rollouts();
        first_step_ = false;
    }

    observation_set_ = true;
}

void Solver::copy_observation() {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    x0_internal_ = x0_;
    t0_internal_ = reset_time_;
}

void Solver::initialize_rollouts() {
    std::shared_lock<std::shared_mutex> lock_state(state_mutex_);
    opt_roll_.clear();
    std::fill(opt_roll_.uu.begin(), opt_roll_.uu.end(),
              dynamics_->get_zero_input());

    std::shared_lock<std::shared_mutex> lock(rollout_cache_mutex_);
    std::fill(opt_roll_cache_.xx.begin(), opt_roll_cache_.xx.end(),
              x0_internal_);
    std::fill(opt_roll_cache_.uu.begin(), opt_roll_cache_.uu.end(),
              dynamics_->get_zero_input());
    for (int i = 0; i < steps_; i++) {
        opt_roll_cache_.tt[i] = t0_internal_ + config_.step_size * i;
    }
}

void Solver::prepare_rollouts() {
    // cleanup
    for (auto& roll : rollouts_) {
        roll.clear_cost();
        roll.valid = true;
    }
}

void Solver::set_reference_trajectory(mppi::reference_trajectory_t& ref) {
    if (ref.rr.size() != ref.tt.size()) {
        std::stringstream error;
        error << "The reference trajectory state and time dimensions do not "
                 "match: "
              << ref.rr.size() << " != " << ref.tt.size();
        throw std::runtime_error(error.str());
    }
    std::unique_lock<std::shared_mutex> lock(reference_mutex_);
    rr_tt_ref_ = ref;
    reference_set_ = true;
}

void Solver::update_reference() {
    std::shared_lock<std::shared_mutex> lock(reference_mutex_);
    cost_->set_reference_trajectory(rr_tt_ref_);

    if (config_.threads > 1) {
        for (auto& cost : cost_v_) {
            cost->set_reference_trajectory(rr_tt_ref_);
            cost->local_track_mode = cost_->local_track_mode;
            cost->base_aim = cost_->base_aim;
            cost->last_u = cost_->last_u;
        }
    }
}

void Solver::sample_trajectories() {
    policy_->shift(t0_internal_);
    policy_->update_samples(weights_, cached_rollouts_);

    if (config_.threads == 1) {
        sample_trajectories_batch(dynamics_, cost_, 0, config_.rollouts);
    } else {
        for (unsigned int i = 0; i < config_.threads; i++) {
            futures_[i] = pool_->enqueue(
                std::bind(&Solver::sample_trajectories_batch, this,
                          std::placeholders::_1, std::placeholders::_2,
                          std::placeholders::_3, std::placeholders::_4),
                dynamics_v_[i], cost_v_[i],
                (unsigned int)i * config_.rollouts / config_.threads,
                (unsigned int)(i + 1) * config_.rollouts / config_.threads);
        }

        for (unsigned int i = 0; i < config_.threads; i++) {
            futures_[i].get();
        }
    }
}

void Solver::sample_trajectories_batch(dynamics_ptr& dynamics, cost_ptr& cost,
                                       const unsigned int start_idx,
                                       const unsigned int end_idx) {
    observation_t x;

    for (unsigned int k = start_idx; k < end_idx; k++) {
        dynamics->reset(x0_internal_, t0_internal_);
        x = x0_internal_;
        double ts;
        for (int t = 0; t < steps_; t++) {
            ts = t0_internal_ + t * config_.step_size;
            rollouts_[k].tt[t] = ts;
            rollouts_[k].uu[t] = policy_->sample(ts, k);

            // compute input-state stage cost
            double cost_temp;
            cost_temp = std::pow(config_.discount_factor, t) *
                        cost->get_stage_cost(x, rollouts_[k].uu[t], ts);
            // store data
            rollouts_[k].xx[t] = x;
            rollouts_[k].cc(t) = cost_temp;
            rollouts_[k].total_cost += cost_temp;

            // integrate dynamics
            x = dynamics->step(rollouts_[k].uu[t], config_.step_size);
        }
    }
}

void Solver::compute_weights() {
    // keep all non diverged rollouts
    double min_cost = std::numeric_limits<double>::max();
    double max_cost = -min_cost;

    for (unsigned int k = 0; k < config_.rollouts; k++) {
        if (rollouts_[k].valid) {
            const double& cost = rollouts_[k].total_cost;
            min_cost = (cost < min_cost) ? cost : min_cost;
            max_cost = (cost > max_cost) ? cost : max_cost;
        }
    }

    min_cost_ = min_cost;
    max_cost_ = max_cost;

    double sum = 0.0;
    for (unsigned int k = 0; k < config_.rollouts; k++) {
        double modified_cost = config_.h *
                               (rollouts_[k].total_cost - min_cost_) /
                               (max_cost_ - min_cost_);

        weights_[k] = rollouts_[k].valid ? std::exp(-modified_cost) : 0.0;
        sum += weights_[k];
    }
    std::transform(weights_.begin(), weights_.end(), weights_.begin(),
                   [&sum](double v) -> double { return v / sum; });
}

void Solver::optimize() {
    // get new rollouts weights
    compute_weights();

    // update policy according to new weights
    policy_->update(weights_, config_.alpha);

    // retrieve the nominal policy for each time step
    for (int t = 0; t < steps_; t++) {
        opt_roll_.tt[t] = t0_internal_ + t * config_.step_size;
        opt_roll_.uu[t] =
            policy_->nominal(t0_internal_ + t * config_.step_size);
    }
}

void Solver::filter_input() {
    // reset the dynamics such that we rollout the dynamics again
    // with the input we filter step after step (sequentially)
    dynamics_->reset(x0_internal_, t0_internal_);
    opt_roll_.xx[0] = x0_internal_;

    // sequential filtering otherwise just nominal rollout
    for (int t = 0; t < steps_ - 1; t++) {
        opt_roll_.xx[t + 1] =
            dynamics_->step(opt_roll_.uu[t], config_.step_size);
    }
}

void Solver::get_input(const observation_t& x, input_t& u, const double t) {
    static double coeff;
    static unsigned int idx;
    {
        std::shared_lock<std::shared_mutex> lock(rollout_cache_mutex_);
        if (t < opt_roll_cache_.tt.front()) {
            std::stringstream warning;
            warning << "Queried time " << t
                    << " smaller than first available time "
                    << opt_roll_cache_.tt.front();
            ROS_ERROR_STREAM(warning.str());
            u = dynamics_->get_zero_input();
        }

        auto lower = std::lower_bound(opt_roll_cache_.tt.begin(),
                                      opt_roll_cache_.tt.end(), t);
        if (lower == opt_roll_cache_.tt.end()) {
            std::stringstream warning;
            warning << "Queried time " << t
                    << " larger than last available time "
                    << opt_roll_cache_.tt.back();
            ROS_ERROR_STREAM(warning.str());
            u = opt_roll_cache_.uu.back();
            return;
        }

        idx = std::distance(opt_roll_cache_.tt.begin(), lower);
        // first index (time)
        if (idx == 0) {
            u = opt_roll_cache_.uu.front();
        }
        // last index (time larget then last step)
        else if (idx > opt_roll_cache_.steps_) {
            u = opt_roll_cache_.uu.back();
        }
        // interpolate
        else {
            coeff = (t - *(lower - 1)) / (*lower - *(lower - 1));
            u = (1 - coeff) * opt_roll_cache_.uu[idx - 1] +
                coeff * opt_roll_cache_.uu[idx];
        }
    }
}

bool Solver::get_optimal_rollout(observation_array_t& xx, input_array_t& uu) {
    std::shared_lock<std::shared_mutex> lock(rollout_cache_mutex_);
    auto lower = std::lower_bound(opt_roll_cache_.tt.begin(),
                                  opt_roll_cache_.tt.end(), reset_time_);
    if (lower == opt_roll_cache_.tt.end()) return false;
    unsigned int offset = std::distance(opt_roll_cache_.tt.begin(), lower);

    // fill with portion of vector starting from current time
    xx = observation_array_t(opt_roll_cache_.xx.begin() + offset,
                             opt_roll_cache_.xx.end());
    uu = input_array_t(opt_roll_cache_.uu.begin() + offset,
                       opt_roll_cache_.uu.end());
    return true;
}

void Solver::swap_policies() {
    std::unique_lock<std::shared_mutex> lock(rollout_cache_mutex_);
    opt_roll_cache_ = opt_roll_;
}
}  // namespace mppi
