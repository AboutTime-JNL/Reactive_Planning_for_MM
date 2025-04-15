#pragma once
#include "core/config.h"
#include "core/cost.h"
#include "core/dynamics.h"
#include "core/policy.h"
#include "core/rollout.h"

#include "utils/savgol_filter.h"
#include "utils/thread_pool.h"

namespace mppi {

class Solver {
   public:
    /**
     * @brief Path Integral Control class
     * @param dynamics: used to forward simulate the system and produce rollouts
     * @param cost: to obtain running cost as function of the stage observation
     * @param policy: implementation of stochastic policy
     * @param config: the solver configuration
     */
    Solver(dynamics_ptr dynamics, cost_ptr cost, policy_ptr policy,
           const config_t& config);
    Solver() = default;
    ~Solver() = default;

   private:
    /**
     * @brief Init the data structures used during the computation
     */
    void init_data();

    /**
     * @brief Initializes the variable for multithreading is required (threads >
     * 1)
     */
    void init_threading();

   public:
    /**
     * @brief Filter the input according to chosen filter
     */
    void filter_input();

    /**
     * @brief Transforms the cost to go into the corresponding sample weights
     * @return
     */
    void compute_weights();

    /**
     * @brief First multiple rollouts are forward simulated and averaged to
     * compute a newly refined control input
     * @param x current observation
     */
    void update_policy();

    /**
     * @brief Sample a batch of trajectories
     * @param dynamics the dynamics function to use
     * @param cost the cost function to use
     * @param start_idx first index in the rollouts vector
     * @param end_idx last index in the rollouts vector
     */
    void sample_trajectories_batch(dynamics_ptr& dynamics, cost_ptr& cost,
                                   const unsigned int start_idx,
                                   const unsigned int end_idx);

    /**
     * @brief Sample multiple trajectories starting from current observation and
     * applying noise input with mean the previous shifted optimal input. A
     * ratio of best previous rollouts is reused to warm start new sample
     * generation and one rollout is a noise free one.
     */
    void sample_trajectories();

    /**
     * @brief Use the data collected from rollouts and return the optimal input
     */
    void optimize();

    /**
     * @brief Prepare rollouts before the new optimization starts.
     * @details The previous optimal rollout as well a portion of cached
     * rollouts (according to the caching_factor) get trimmed up to the new
     * reset time. The remaining are cleared to be overwritten in the next
     * control loop. Cached rollouts has different additive noise wrt new
     * updated optimal input, thus this needs to be recomputed
     */
    void prepare_rollouts();

    /**
     * @brief Initializes rollouts for the first time
     */
    void initialize_rollouts();

    /**
     * @brief Fill the optimized policy cache with the latest policy and set
     * flags
     */
    void swap_policies();

    /**
     * @brief Return the latest optimal input for the current time and
     * observation
     * @param x[in, unused]: current observation
     * @param u[out]: optimal input
     * @param t[int]: current time
     */
    void get_input(const observation_t& x, input_t& u, const double t);

    /**
     * @brief Returns the optimal rollout from the latest updated time when
     * calling set_observation
     * @param x the state trajectory
     * @param u the input trajectory
     * @return false if time is later then the latest available in the nominal
     * rollout
     */
    bool get_optimal_rollout(observation_array_t& x, input_array_t& u);

    /**
     * @brief Reset the initial observation (state) and time for next
     * optimization
     * @param x: current observation
     * @param t: current time
     */
    void set_observation(const observation_t& x, const double t);

    /**
     * @brief Copy latest observation and time to internal variables used for
     * the next optimization
     * @details Copying latest x and t to internal variables is required to
     * perform the next optimization starting from the same initial state and
     * time while the public one can be potentially changed by a different
     * thread.
     */
    void copy_observation();
    /**
     * @brief Set the reference of the cost function to the latest received for
     * the new optimization
     */
    void update_reference();

   public:
    /**
     * @brief Set the reference trajectory to be used in the next optimization
     * loop
     * @param ref: reference trajectory
     */
    void set_reference_trajectory(reference_trajectory_t& ref);

    // TODO clean this up: everything public...
   public:
    std::vector<double> weights_;

    dynamics_ptr dynamics_;
    cost_ptr cost_;
    policy_ptr policy_;
    config_t config_;

    unsigned int cached_rollouts_;

    std::vector<Rollout> rollouts_;
    int steps_;

    bool first_step_ = true;

    // local minima mode
    bool local_minima_mode = false;
    Eigen::Vector2d last_base_point;
    std::vector<Eigen::Vector2d> base_path;

   protected:
    // time from which the current optimization has started
    double reset_time_;
    // first state for simulation
    observation_t x0_;
    // internal t0 for next optimization
    double t0_internal_;
    // internal x0 for next optimization
    observation_t x0_internal_;
    // state dimension
    unsigned int nx_;
    // input dimension
    unsigned int nu_;
    // optimized rollout
    Rollout opt_roll_;
    // previously optimized rollout
    Rollout opt_roll_cache_;
    // min cost among all rollouts
    double min_cost_;
    // max cost among all rollouts
    double max_cost_;
    // flag to check that observation has ever been set
    std::atomic_bool observation_set_;
    // protects access to the solution
    std::shared_mutex rollout_cache_mutex_;
    // protects access to the state
    std::shared_mutex state_mutex_;
    // flag to check that reference has ever been set
    std::atomic_bool reference_set_;
    // protects access to the reference trajectory
    std::shared_mutex reference_mutex_;
    // reference used during optimization
    reference_trajectory_t rr_tt_ref_;
    // thread pool
    std::unique_ptr<ThreadPool> pool_;
    // futures results from the thread pool
    std::vector<std::future<void>> futures_;
    // vector of dynamics functions used per each thread
    std::vector<dynamics_ptr> dynamics_v_;
    // vector of cost functions used per each thread
    std::vector<cost_ptr> cost_v_;
};

}  // namespace mppi
