#pragma once
#include "core/config.h"
#include "core/typedefs.h"
#include "utils/multivariate_normal_eigen.h"
#include "utils/savgol_filter.h"

namespace mppi {

/// Generic policy class for implementing a policy
/// Here the policy is intended as stochastic, therefore this class implements
/// also the methods to generate samples out of a nominal policy. These can be
/// used by the controller to generate rollouts
class Policy {
   public:
    Policy(int nu, const Config& config);

    /**
     * Update the samples using "performance" weights assigned to each of them.
     * The parameter keep can be used to tell the policy to keep the best n out
     * of all the samples (according to the associated weights)
     */
    void update_samples(const std::vector<double>& weights, const int keep);

    /**
     * Update the policy, given a set of sample weights and a step size for the
     * gradient step
     */
    void update(const std::vector<double>& weights, const double step_size);

    /**
     * Shift the policy to the new time t
     */
    void shift(const double t);

    /**
     * Return the nominal policy
     */
    Eigen::VectorXd nominal(double t);

    /**
     * Return a single sample
     */
    Eigen::VectorXd sample(double t, int k);

    /**
     * Bound the input to the predefined input bounds
     */
    void bound();

   protected:
    int nu_;

   private:
    Config config_;
    double dt_;
    int ns_;
    int nt_;
    Eigen::ArrayXd t_;
    std::shared_ptr<multivariate_normal> dist_;
    std::vector<Eigen::MatrixXd> samples_;
    Eigen::MatrixXd nominal_;
    Eigen::MatrixXd delta_;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
        L_;  // matrix for shift operation of all the samples

    SavGolFilter filter_;

    Eigen::MatrixXd gradient_;
    Eigen::MatrixXd max_gradient_;

    Eigen::MatrixXd max_limits_;
    Eigen::MatrixXd min_limits_;
};

}  // namespace mppi