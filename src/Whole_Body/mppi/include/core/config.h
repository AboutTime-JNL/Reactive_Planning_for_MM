#pragma once
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <iostream>
#include <optional>
#include <string>

namespace mppi {

/**
  Structure with all the configuration parameters for the MPPI solver
**/
struct Config {
    // solver parameters
    unsigned int rollouts = 1;
    double lambda = 1.0;
    double h = 1.0;
    double step_size = 0.1;
    double horizon = 1.0;

    double caching_factor = 0.0;
    unsigned int substeps = 1;

    double alpha = 1.0;
    Eigen::VectorXd input_variance;

    double discount_factor = 1.0;
    unsigned int threads = 1;

    bool filtering = false;
    std::vector<uint> filters_order;
    std::vector<int> filters_window;

    Eigen::VectorXd u_min;
    Eigen::VectorXd u_max;

    // controller parameters
    double linear_weight = 0.0;
    double angular_weight = 0.0;
    double safe_distance = 0.0;

    double policy_update_rate = 0.0;
    double reference_update_rate = 0.0;
    double esdf_update_rate = 0.0;

    double sim_dt = 0.0;

    bool init_from_file(const std::string& file);

   private:
    template <typename T>
    std::optional<T> parse_key(const YAML::Node& node, const std::string& key);
};

template <typename T>
std::optional<T> Config::parse_key(const YAML::Node& node,
                                   const std::string& key) {
    if (!node[key]) {
        std::cout << "Could not find entry: " << key << std::endl;
        return {};
    }
    return node[key].as<T>();
};

template <>
inline std::optional<Eigen::VectorXd> Config::parse_key<Eigen::VectorXd>(
    const YAML::Node& node, const std::string& key) {
    if (!node[key]) {
        std::cout << "Could not find entry: " << key << std::endl;
        return {};
    }
    auto v = node[key].as<std::vector<double>>();
    Eigen::VectorXd v_eigen(v.size());
    for (unsigned int i = 0; i < v.size(); i++) v_eigen(i) = v[i];

    return v_eigen;
};

}  // namespace mppi

std::ostream& operator<<(std::ostream& os, const mppi::Config& config);