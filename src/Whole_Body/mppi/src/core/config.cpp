#include <cmath>
#include <iostream>

#include "core/config.h"

using namespace mppi;

bool Config::init_from_file(const std::string& file) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(file);
    } catch (const YAML::ParserException& ex) {
        std::cout << ex.what() << std::endl;
    } catch (const YAML::BadFile& ex) {
        std::cout << ex.what() << std::endl;
    }

    YAML::Node solver_options = config["solver"];
    if (!solver_options) {
        std::cout << "Failed to parse solver options." << std::endl;
        return false;
    }

    // clang-format off
    rollouts        = parse_key<unsigned int>(solver_options, "rollouts").value_or(rollouts);
    lambda          = parse_key<double>(solver_options, "lambda").value_or(lambda);
    h               = parse_key<double>(solver_options, "h").value_or(h);
    step_size       = parse_key<double>(solver_options, "step_size").value_or(step_size);
    horizon         = parse_key<double>(solver_options, "horizon").value_or(horizon);
    caching_factor  = parse_key<double>(solver_options, "caching_factor").value_or(caching_factor);
    substeps        = parse_key<unsigned int>(solver_options, "substeps").value_or(substeps);
    alpha           = parse_key<double>(solver_options, "gradient_step_size").value_or(alpha);
    input_variance  = parse_key<Eigen::VectorXd>(solver_options, "input_variance").value_or(Eigen::VectorXd(0));
    discount_factor = parse_key<double>(solver_options, "discount_factor").value_or(discount_factor);
    threads         = parse_key<int>(solver_options, "threads").value_or(threads);
    filtering       = parse_key<bool>(solver_options, "filtering").value_or(filtering);
    filters_order   = parse_key<std::vector<uint>>(solver_options, "filters_order").value_or(std::vector<uint>{});
    filters_window  = parse_key<std::vector<int>>(solver_options, "filters_window").value_or(std::vector<int>{});
    u_min           = parse_key<Eigen::VectorXd>(solver_options, "u_min").value_or(Eigen::VectorXd(0));
    u_max           = parse_key<Eigen::VectorXd>(solver_options, "u_max").value_or(Eigen::VectorXd(0));

    linear_weight   = parse_key<double>(config, "linear_weight").value_or(linear_weight);
    angular_weight  = parse_key<double>(config, "angular_weight").value_or(angular_weight);
    safe_distance = parse_key<double>(config, "safe_distance").value_or(safe_distance);
    policy_update_rate      = parse_key<double>(config, "policy_update_rate").value_or(policy_update_rate);
    reference_update_rate   = parse_key<double>(config, "reference_update_rate").value_or(reference_update_rate);
    esdf_update_rate        = parse_key<double>(config, "esdf_update_rate").value_or(esdf_update_rate);
    sim_dt                  = parse_key<double>(config, "sim_dt").value_or(sim_dt);
    //clang-format on
  
    std::cout << "Solver options correctly parsed from: " << file << std::endl;
    return true;
}