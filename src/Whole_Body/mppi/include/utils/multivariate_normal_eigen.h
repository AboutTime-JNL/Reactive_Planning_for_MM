#pragma once

#include <Eigen/Dense>
#include <random>

namespace mppi {

// Implementation from the following post
// https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c

/**
 * @brief Utitlity class to perform sampling from a multivariate normal
 * distribution
 */
struct multivariate_normal {
    multivariate_normal(Eigen::MatrixXd const& covar)
        : multivariate_normal(Eigen::VectorXd::Zero(covar.rows()), covar) {}

    multivariate_normal(Eigen::VectorXd const& mean,
                        Eigen::MatrixXd const& covar)
        : mean(mean) {
        set_covariance(covar);
    }

    void set_covariance(Eigen::MatrixXd const& cov) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
        transform = eigenSolver.eigenvectors() *
                    eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const {
        static std::mt19937 gen{std::random_device{}()};
        static std::normal_distribution<double> dist;

        return mean + transform * Eigen::VectorXd{mean.size()}.unaryExpr(
                                      [&](auto x) { return dist(gen); });
    }

    // Sample each column in the matrix from a multivariate normal
    void setRandomRow(Eigen::Ref<Eigen::MatrixXd> m) {
        for (int i = 0; i < m.rows(); i++) m.row(i) = (*this)();
    }
};

}  // namespace mppi