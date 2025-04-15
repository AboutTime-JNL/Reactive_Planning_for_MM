#pragma once
#include <core/solver.h>

#include <ros/callback_queue.h>
#include <ros/node_handle.h>
#include <threading/WorkerManager.hpp>

#include "core/model.h"

#include <std_msgs/Float64MultiArray.h>
#include "Fiesta.h"

#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/JointState.h>
#include "std_msgs/Float64MultiArray.h"

using namespace mppi;

namespace mppi {

class Controller {
   public:
    Controller() = delete;
    Controller(ros::NodeHandle& nh, ros::NodeHandle& nh_sensor,
               ros::NodeHandle& nh_local);
    ~Controller();

    /**
     * @brief Starts all the threads
     * @return
     */
    bool start();

    /**
     * @brief Stop the running controller and associated threads
     */
    void stop();

    /**
     * @brief         set last estimated state
     * TODO(giuseppe) this is dangerous since one might not use correctly this
     * class; split a sync vs an async class
     *
     * @param         x
     * @param         t
     * @attention
     */
    void set_observation(const observation_t& x, const double& t);

    /**
     * @brief         only feedforward term required
     *
     * @param         x
     * @param         u
     * @param         t
     * @attention
     */
    void get_input(const observation_t& x, input_t& u, const double& t);

    void set_ee_pos(mppi::Pose target);

    void set_joint_state(const observation_t q);

   private:
    bool init_ros();

    void base_path_callback(const std_msgs::Float64MultiArray::ConstPtr& msg);

    void aim_joint_state_callback(const sensor_msgs::JointStateConstPtr& msg);

    void local_minima_detect();

    bool update_policy_thread(const mppi::threading::WorkerEvent& event);
    bool update_reference_thread(const mppi::threading::WorkerEvent& event);
    bool update_esdf_thread(const mppi::threading::WorkerEvent& event);

   public:
    mppi::Config config_;
    Eigen::VectorXd x_;
    bool game_start_ = false;
    bool game_reset_ = false;
    bool esdf_reset_ = false;
    mppi::observation_t aim_joint_state_;
    mppi::observation_t start_joint_state_;

    std::shared_ptr<fiesta::Fiesta<sensor_msgs::PointCloud2::ConstPtr,
                                   geometry_msgs::TransformStamped::ConstPtr>>
        esdf_map_sensor_ = nullptr;

   private:
    mppi::threading::WorkerManager worker_manager_;

    std::mutex model_mutex_;
    mppi::RobotModel robot_model_;

    // state flags
    bool started_;
    bool initialized_;
    bool observation_set_;

    // controller components
    mppi::solver_ptr solver_;
    mppi::cost_ptr cost_;
    mppi::policy_ptr policy_;
    mppi::dynamics_ptr dynamics_;

    // ros node handles
    ros::NodeHandle nh_;
    ros::NodeHandle nh_sensor_;
    ros::CallbackQueue sensor_queue;
    ros::NodeHandle nh_local_;
    ros::CallbackQueue local_queue;

    // time record
    double policy_update_time_ = 0.0;
    double esdf_update_time_ = 0.0;

    // success rate
    ros::Subscriber aim_joint_state_subscriber_;
    bool update_reference_ = false;

    // local minima
    ros::Subscriber base_path_subscriber_;
    mppi::input_array_t u_opt_;
    int local_minima_count_ = 0;
    double local_minima_dist;
    int local_cold_time_ = 0;

    // set desired pose
    std::mutex reference_mutex_;
    mppi::reference_trajectory_t ref_;
};
}  // namespace mppi
