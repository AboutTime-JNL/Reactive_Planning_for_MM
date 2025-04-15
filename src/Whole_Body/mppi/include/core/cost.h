#pragma once
#include "core/model.h"

#include <ros/ros.h>
#include "core/typedefs.h"

#include <geometry_msgs/TransformStamped.h>
#include "Fiesta.h"

namespace mppi {

class Cost {
   public:
    Cost(const std::string &robot_description, const double linear_weight,
         const double angular_weight, const double safe_distance,
         ros::NodeHandle &nh);
    ~Cost() = default;

    cost_ptr create();
    cost_ptr clone() const;

    /**
     * @brief Compute the current stage cost
     * @param x: stage observation
     * @param time: stage time
     * @return the stage cost
     */
    cost_t get_stage_cost(const observation_t &x, const input_t &u,
                          const double t = 0);

    /**
     * @brief Set the reference trajectory (optionally) used in the cost
     * function
     * @param ref_traj the new timed reference trajectory
     */
    void set_reference_trajectory(const reference_trajectory_t &traj);

   private:
    /**
     * @brief The derived class must implement this in order to compute the cost
     * at the current time wrt to the given reference
     * @param x current observation
     * @param x_ref reference state extracted from reference trajectory
     * @param t current time
     * @return
     */
    cost_t compute_cost(const observation_t &x, const input_t &u,
                        const reference_t &ref);

    /**
     * @brief Get the reference point closest to the current time.
     * @details Always the next point in time is considered in the default
     * implementation. Different types of interpolation could be implemented by
     * the derived class
     * @param[in]  x: current observation
     * @param[out] ref : the returned reference point
     * @param[in]  t : the query time
     */
    void interpolate_reference(const observation_t &x, reference_t &ref,
                               const double t);

   public:
    mppi::RobotModel robot_model_;

    std::shared_ptr<fiesta::Fiesta<sensor_msgs::PointCloud2::ConstPtr,
                                   geometry_msgs::TransformStamped::ConstPtr>>
        esdf_map_ = nullptr;

    bool local_track_mode = false;
    Eigen::Vector2d base_aim;

    Eigen::VectorXd last_u;

    void publish_sphere(const observation_t x);
    ros::Publisher sphere_pub_;

   private:
    std::string robot_description_;

    double linear_weight_;
    double angular_weight_;
    double safe_distance_;
    ros::NodeHandle nh_;

    // pose error weights
    Eigen::Matrix<double, 3, 3> Q_linear_;
    Eigen::Matrix<double, 3, 3> Q_angular_;

    // collision test
    double Q_obst_ = 10000;

    std::string link_name[8] = {"base",
                                "jaka_base_link",
                                "jaka_shoulder_link",
                                "jaka_upper_arm_link",
                                "jaka_forearm_link",
                                "jaka_wrist_1_link",
                                "jaka_wrist_2_link",
                                "jaka_wrist_3_link"};
    double sphere_num[7] = {1, 1, 1, 2, 3, 1, 2};
    int sphere_row[7] = {0, 1, 2, 3, 5, 8, 9};
    double sphere_r[7][4] = {{0.35, 0., 0., 0.},     {0.05, 0., 0., 0.},
                             {0.05, 0., 0., 0.},     {0.05, 0.05, 0., 0.},
                             {0.05, 0.05, 0.05, 0.}, {0.05, 0., 0., 0.},
                             {0.05, 0.05, 0., 0.}};

    Eigen::Vector3d sphere_pos[7][4];

    // single reference point in time
    reference_t r_;

    // timed sequence of references
    reference_trajectory_t timed_ref_;

    // preference for the manipulator angle
    Eigen::Matrix<double, 6, 1> pos_prefer;
};
}  // namespace mppi
