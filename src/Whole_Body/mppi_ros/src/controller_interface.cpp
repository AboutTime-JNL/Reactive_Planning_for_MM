#include <ros/package.h>
#include <chrono>
#include <fstream>

#include "core/cost.h"
#include "core/dynamics.h"
#include "core/policy.h"
#include "ros_params.h"

#include <ros_params.h>
#include "controller_interface.h"

using namespace mppi;

Controller::Controller(ros::NodeHandle &nh, ros::NodeHandle &nh_sensor,
                       ros::NodeHandle &nh_local)
    : nh_(nh), nh_sensor_(nh_sensor), nh_local_(nh_local) {
    // init params
    std::string robot_description;
    getString(nh_, "/robot_description", robot_description);

    std::string config_file =
        ros::package::getPath("mppi_ros") + "/config/params.yaml";
    if (!config_.init_from_file(config_file)) {
        ROS_ERROR_STREAM("Failed to init solver options from " << config_file);
    }

    // init variables
    nh_sensor_.setCallbackQueue(&sensor_queue);
    nh_local_.setCallbackQueue(&local_queue);
    init_ros();

    x_ = Eigen::VectorXd::Zero(mppi::Dim::STATE_DIMENSION);
    aim_joint_state_ = Eigen::VectorXd::Zero(mppi::Dim::STATE_DIMENSION);
    start_joint_state_ = Eigen::VectorXd::Zero(mppi::Dim::STATE_DIMENSION);

    // reference trajectory
    ref_.rr.resize(1, mppi::observation_t::Zero(Dim::REFERENCE_DIMENSION));
    ref_.tt.resize(1, 0.0);

    // mppi components
    robot_model_.init_from_xml(robot_description);

    dynamics_ = std::make_shared<mppi::Dynamics>();

    esdf_map_sensor_ = std::make_shared<
        fiesta::Fiesta<sensor_msgs::PointCloud2::ConstPtr,
                       geometry_msgs::TransformStamped::ConstPtr>>(nh_sensor_);
    cost_ = std::make_shared<mppi::Cost>(
        robot_description, config_.linear_weight, config_.angular_weight,
        config_.safe_distance, nh_);
    cost_->esdf_map_ = esdf_map_sensor_;

    policy_ = std::make_shared<mppi::Policy>(int(mppi::Dim::INPUT_DIMENSION),
                                             config_);

    solver_ =
        std::make_shared<mppi::Solver>(dynamics_, cost_, policy_, config_);

    initialized_ = true;
    started_ = false;
    ROS_INFO("Controller interface initialized.");
}

Controller::~Controller() { this->stop(); }

bool Controller::init_ros() {
    // local minima subscriber
    base_path_subscriber_ = nh_local_.subscribe(
        "/base_path", 1, &Controller::base_path_callback, this);

    // success rate pose service
    aim_joint_state_subscriber_ = nh_.subscribe(
        "/aim_joint_states", 10, &Controller::aim_joint_state_callback, this);

    ROS_INFO("ROS initialized.");
    return true;
}

void Controller::base_path_callback(
    const std_msgs::Float64MultiArray::ConstPtr &msg) {
    std::cout << "base path size: " << msg->layout.data_offset << std::endl;

    Eigen::Vector2d base_pos;
    base_pos << 0, 0;

    std::vector<Eigen::Vector2d> path;
    path.clear();

    for (int i = msg->layout.data_offset; i > 0; i--) {
        base_pos[0] = msg->data[i * 2 - 2];
        base_pos[1] = msg->data[i * 2 - 1];
        path.push_back(base_pos);
    }

    // 更新轨迹
    if (path.size() > 1) {
        solver_->base_path = path;
        solver_->last_base_point = solver_->base_path.back();
        solver_->base_path.pop_back();
        std::cout << "update base path" << std::endl;
    } else if (path.size() == 1) {
        solver_->base_path = path;
        solver_->last_base_point << x_(0), x_(1);
    }
}

void Controller::aim_joint_state_callback(
    const sensor_msgs::JointStateConstPtr &msg) {
    if (game_start_) {
        game_reset_ = true;
        esdf_reset_ = true;
    }
    game_start_ = true;
    for (int i = 0; i < 9; i++) aim_joint_state_[i] = msg->position[i];
    for (int j = 9; j < 18; j++) start_joint_state_[j - 9] = msg->position[j];

    set_joint_state(aim_joint_state_);
}

void Controller::set_joint_state(mppi::observation_t q) {
    std::unique_lock<std::mutex> lock(model_mutex_);
    robot_model_.update_state(q);
    mppi::Pose ee_pose = robot_model_.get_pose("panda_hand");

    set_ee_pos(ee_pose);
}

void Controller::set_ee_pos(mppi::Pose target) {
    std::unique_lock<std::mutex> lock(reference_mutex_);

    Eigen::VectorXd pr = Eigen::VectorXd::Zero(7);
    pr(0) = target.translation(0);
    pr(1) = target.translation(1);
    pr(2) = target.translation(2);
    pr(3) = target.rotation.x();
    pr(4) = target.rotation.y();
    pr(5) = target.rotation.z();
    pr(6) = target.rotation.w();
    ref_.rr[0].head<7>() = pr;

    update_reference_ = true;
}

void Controller::set_observation(const mppi::observation_t &x,
                                 const double &t) {
    x_ = x;
    solver_->set_observation(x, t);
    observation_set_ = true;
}

void Controller::get_input(const mppi::observation_t &x, mppi::input_t &u,
                           const double &t) {
    solver_->get_input(x, u, t);
}

bool Controller::start() {
    if (!initialized_) {
        ROS_ERROR_STREAM(
            "The controller is not initialized. Have you called the init() "
            "method?");
        return false;
    }

    if (started_) {
        ROS_WARN_STREAM("The controller has already been started.");
        return true;
    }

    esdf_map_sensor_->update_lower_bound << x_(0) - 1, x_(1) - 1, 0;
    esdf_map_sensor_->update_upper_bound << x_(0) + 1, x_(1) + 1, 1.6;

    mppi::threading::WorkerOptions update_reference_opt;
    update_reference_opt.name_ = "update_reference_thread";
    update_reference_opt.timeStep_ = 1.0 / config_.reference_update_rate;
    update_reference_opt.callback_ = std::bind(
        &Controller::update_reference_thread, this, std::placeholders::_1);
    worker_manager_.addWorker(update_reference_opt, true);

    mppi::threading::WorkerOptions update_policy_opt;
    update_policy_opt.name_ = "update_policy_thread";
    update_policy_opt.timeStep_ = (config_.policy_update_rate == 0)
                                      ? 0
                                      : 1.0 / config_.policy_update_rate;
    update_policy_opt.callback_ = std::bind(&Controller::update_policy_thread,
                                            this, std::placeholders::_1);
    worker_manager_.addWorker(update_policy_opt, true);

    mppi::threading::WorkerOptions update_esdf_opt;
    update_esdf_opt.name_ = "update_esdf_thread";
    update_esdf_opt.timeStep_ =
        (config_.esdf_update_rate == 0) ? 0 : 1.0 / config_.esdf_update_rate;
    update_esdf_opt.callback_ =
        std::bind(&Controller::update_esdf_thread, this, std::placeholders::_1);
    worker_manager_.addWorker(update_esdf_opt, true);

    started_ = true;
    return true;
}

void Controller::stop() { worker_manager_.stopWorkers(); }

void Controller::local_minima_detect() {
    if (!ref_.rr.empty() && !u_opt_.empty()) {
        Eigen::Vector2d pos;
        Eigen::Vector2d error;

        pos << x_[0], x_[1];
        error = pos - ref_.rr[0].head<2>();

        if (error.norm() > 0.5) {
            if (!solver_->local_minima_mode) {
                // check if the optimal trajectory is in local minima
                int j = 0;
                for (int i = 0; i < 5; i++) {
                    if (u_opt_[i].norm() < 0.1) j++;
                }
                if (j > 2)
                    local_minima_count_++;
                else
                    local_minima_count_ = 0;

                if (local_minima_count_ > 10 && local_cold_time_ > 10) {
                    std::cout << "find local minima" << std::endl;
                    solver_->local_minima_mode = true;
                    local_minima_count_ = 0;

                    // record the local minima distance
                    Eigen::Vector3d error_ee;
                    std::unique_lock<std::mutex> lock(model_mutex_);
                    robot_model_.update_state(x_);
                    mppi::Pose ee_pose = robot_model_.get_pose("panda_hand");
                    error_ee = ee_pose.translation - ref_.rr[0].head<3>();

                    local_minima_dist = error_ee.norm();
                    std::cout << "local minima dist: " << local_minima_dist
                              << std::endl;
                }

                local_cold_time_++;
            } else {
                local_queue.callAvailable(ros::WallDuration());

                // check if the robot is out of local minima
                Eigen::Vector3d error_ee;
                std::unique_lock<std::mutex> lock(model_mutex_);
                robot_model_.update_state(x_);
                mppi::Pose ee_pose = robot_model_.get_pose("panda_hand");
                error_ee = ee_pose.translation - ref_.rr[0].head<3>();

                if (error_ee.norm() < local_minima_dist - 0.5) {
                    solver_->local_minima_mode = false;
                    local_cold_time_ = 0;
                    std::cout << "out of local minima" << std::endl;
                }
            }
        } else {
            if (solver_->local_minima_mode) {
                solver_->local_minima_mode = false;
                local_cold_time_ = 0;
                std::cout << "out of local minima" << std::endl;
            }
        }
    }
}

bool Controller::update_policy_thread(
    const mppi::threading::WorkerEvent &event) {
    if (!observation_set_) return true;

    if (esdf_reset_) {
        esdf_map_sensor_->Reset();
        esdf_reset_ = false;
    }

    solver_->update_policy();

    local_minima_detect();

    return true;
}

bool Controller::update_reference_thread(
    const mppi::threading::WorkerEvent &event) {
    std::unique_lock<std::mutex> lock(reference_mutex_);

    if (update_reference_) {
        update_reference_ = false;
        solver_->set_reference_trajectory(ref_);
    }

    return true;
}

bool Controller::update_esdf_thread(const mppi::threading::WorkerEvent &event) {
    mppi::observation_array_t mean_x;
    solver_->get_optimal_rollout(mean_x, u_opt_);

    if (esdf_reset_) {
        return true;
    }

    if (!mean_x.empty()) {
        Eigen::VectorXd x_base(mean_x.size() * 2);
        Eigen::VectorXd y_base(mean_x.size() * 2);
        x_base.setZero();
        y_base.setZero();

        for (int i = 0; i < mean_x.size(); i++) {
            double range = std::sqrt(std::pow(0.25, 2) * i) * 0.1;
            x_base(i * 2) = mean_x[i](0) + range;
            y_base(i * 2) = mean_x[i](1) + range;
            x_base(i * 2 + 1) = mean_x[i](0) - range;
            y_base(i * 2 + 1) = mean_x[i](1) - range;
        }

        esdf_map_sensor_->update_lower_bound << x_base.minCoeff() - 1,
            y_base.minCoeff() - 1, 0;
        esdf_map_sensor_->update_upper_bound << x_base.maxCoeff() + 1,
            y_base.maxCoeff() + 1, 1.6;
    }

    sensor_queue.callAvailable(ros::WallDuration());

    esdf_map_sensor_->UpdateEsdfEvent();

    return true;
}
