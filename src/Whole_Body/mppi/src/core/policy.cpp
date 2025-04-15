#include "core/policy.h"
#include "utils/utils.h"

using namespace mppi;

const double epsilon = 1e-9;

Policy::Policy(int nu, const Config& config) : nu_(nu), config_(config) {
    Eigen::MatrixXd C = config.input_variance.asDiagonal();
    dist_ = std::make_shared<multivariate_normal>(C);
    dt_ = config.step_size;
    ns_ = config.rollouts;
    nt_ = static_cast<int>(std::ceil(config.horizon / dt_));
    samples_.resize(ns_, Eigen::MatrixXd::Zero(nt_, nu));

    t_ = Eigen::ArrayXd::LinSpaced(nt_, 0.0, nt_ - 1) * dt_;
    nominal_ = Eigen::MatrixXd::Zero(nt_, nu);
    delta_ = Eigen::MatrixXd::Zero(nt_, nu);
    gradient_ = Eigen::MatrixXd::Zero(nt_, nu);

    L_.setIdentity(nt_);

    // Initialize filter
    if (config_.filtering) {
        filter_ = SavGolFilter(nt_, nu_, config.filters_window,
                               config.filters_order, 0, dt_);
    }

    // initialize limits
    max_limits_ = Eigen::MatrixXd::Ones(nt_, nu_);
    min_limits_ = -Eigen::MatrixXd::Ones(nt_, nu_);

    // used to clip the gradient
    max_gradient_ = Eigen::MatrixXd::Ones(nt_, nu_) * 0.2;

    for (int i = 0; i < nt_; i++) {
        max_limits_.row(i) = config.u_max;
        min_limits_.row(i) = config.u_min;
    }
}

Eigen::VectorXd Policy::nominal(double t) {
    auto it =
        std::upper_bound(t_.data(), t_.data() + t_.size(), t,
                         [](double a, double b) { return (b - a) > epsilon; });
    unsigned int time_idx = std::distance(t_.data(), it) - 1;

    double alpha = (t - t_(time_idx)) / (t_(time_idx + 1) - t_(time_idx));
    return (1 - alpha) * nominal_.row(time_idx) +
           alpha * nominal_.row(time_idx + 1);
}

Eigen::VectorXd Policy::sample(double t, int k) {
    auto it =
        std::upper_bound(t_.data(), t_.data() + t_.size(), t,
                         [](double a, double b) { return (b - a) > epsilon; });
    unsigned int time_idx = std::distance(t_.data(), it) - 1;

    Eigen::VectorXd uu = samples_[k].row(time_idx) + nominal_.row(time_idx);
    return uu;
}

void Policy::update_samples(const std::vector<double>& weights,
                            const int keep) {
    if (keep == 0) {
        for (auto& sample : samples_) dist_->setRandomRow(sample);
    } else {
        std::vector<unsigned int> sorted_idxs = sort_indexes(weights);
        for (int i = keep; i < ns_ - 3; i++) {
            dist_->setRandomRow(samples_[sorted_idxs[i]]);
        }

        // noise free sample
        samples_[sorted_idxs[ns_ - 2]].setZero();

        // sample exactly zero velocity
        samples_[sorted_idxs[ns_ - 1]] = -nominal_;
    }

    bound();  // TODO should bound each sample so that a convex combination is
              // also within bounds
}

void Policy::update(const std::vector<double>& weights,
                    const double step_size) {
    delta_ = nominal_;
    gradient_.setZero();

    for (int i = 0; i < ns_; i++) {
        gradient_ += samples_[i] * weights[i];
    }

    // clip gradient
    gradient_ = gradient_.cwiseMax(-max_gradient_).cwiseMin(max_gradient_);

    nominal_ += step_size * gradient_;

    // update filter with new nominal
    if (config_.filtering) {
        filter_.reset(t_[0]);
        for (long int i = 0; i < t_.size(); i++) {
            filter_.add_measurement(nominal_.row(i), t_(i));
        }
        for (long int i = 0; i < t_.size(); i++) {
            filter_.apply(nominal_.row(i), t_(i));
        }
    }

    // update noise to current nominal trajectory
    delta_ = nominal_ - delta_;
    for (auto& sample : samples_) sample -= delta_;
}

void Policy::shift(const double t) {
    static int time_idx_shift;
    if (t < t_[0]) {
        throw std::runtime_error("Shifting back in time!");
    }

    if (t >= t_[1]) {
        time_idx_shift =
            std::distance(t_.data(), std::upper_bound(
                                         t_.data(), t_.data() + t_.size(), t)) -
            1;

        t_ += dt_ * time_idx_shift;

        // 如果已经移动到最后，则固定 `nominal_` 为最后一行的值，并清零
        // `samples_
        if (time_idx_shift == nt_ - 1) {
            Eigen::RowVectorXd last_nominal = nominal_.row(nt_ - 1);
            nominal_.rowwise() = last_nominal;
            for (auto& sample : samples_) sample.setZero();
            return;
        }

        // 构造 `L_` 变换矩阵，使得数据向前移动 `time_idx_shift` 行
        L_.setIdentity();
        std::transform(L_.indices().data(), L_.indices().data() + nt_,
                       L_.indices().data(), [this](int i) -> int {
                           return (i < time_idx_shift)
                                      ? nt_ + i - time_idx_shift
                                      : i - time_idx_shift;
                       });

        for (auto& sample : samples_) {
            sample = L_ * sample;
            sample.bottomLeftCorner(time_idx_shift, nu_).setZero();
        }

        nominal_ = L_ * nominal_;
        nominal_.bottomLeftCorner(time_idx_shift, nu_).setZero();
    }
}

void Policy::bound() {
    for (auto& sample : samples_) {
        sample = sample.cwiseMax(min_limits_).cwiseMin(max_limits_);
    }
}