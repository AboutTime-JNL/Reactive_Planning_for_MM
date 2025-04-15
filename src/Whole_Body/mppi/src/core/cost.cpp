#include "core/cost.h"
#include <cmath>

#include <chrono>
#include <iostream>
#include <thread>

#include <visualization_msgs/MarkerArray.h>

namespace mppi {

Cost::Cost(const std::string& robot_description, const double linear_weight,
           const double angular_weight, const double safe_distance,
           ros::NodeHandle& nh)
    : robot_description_(robot_description),
      linear_weight_(linear_weight),
      angular_weight_(angular_weight),
      safe_distance_(safe_distance),
      nh_(nh) {
    robot_model_.init_from_xml(robot_description);

    Q_linear_ = Eigen::Matrix3d::Identity() * linear_weight;
    Q_angular_ = Eigen::Matrix3d::Identity() * angular_weight;

    esdf_map_ = std::make_shared<
        fiesta::Fiesta<sensor_msgs::PointCloud2::ConstPtr,
                       geometry_msgs::TransformStamped::ConstPtr>>(nh_);

    sphere_pos[0][0] << 0., 0., -0.029;

    sphere_pos[1][0] << 0., 0., 0.12015;

    sphere_pos[2][0] << 0., 0.14415, 0.;

    sphere_pos[3][0] << 0.18, 0., 0.;
    sphere_pos[3][1] << 0.36, 0., 0.;

    sphere_pos[4][0] << 0., 0., 0.;
    sphere_pos[4][1] << 0.15175, 0., 0.;
    sphere_pos[4][2] << 0.3035, 0., 0.;

    sphere_pos[5][0] << 0., 0.1135, 0.;

    sphere_pos[6][0] << 0., 0.107, 0.;
    sphere_pos[6][1] << 0., 0.214, 0.;

    pos_prefer << 0, 1.3094, -1.1732, -4.9579, -3.0013, -3.2900;
    last_u = Eigen::VectorXd::Zero(9);

    sphere_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>("sphere_marker", 1000);
}

/**
 * @brief        用于创建一个新的 Cost 对象，并使用特定的参数进行初始化。
 *
 * @return        cost_ptr
 * @attention
 */
cost_ptr Cost::create() {
    return std::make_shared<Cost>(robot_description_, linear_weight_,
                                  angular_weight_, safe_distance_, nh_);
}

void Cost::publish_sphere(const observation_t x) {
    robot_model_.update_state(x);

    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "sphere_markers";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    marker.lifetime = ros::Duration();

    // obstacle avoidance cost
    Eigen::Vector3d evaluat_sphere;
    mppi::Pose start;

    for (int i = 0; i < 7; i++) {
        mppi::Pose start = robot_model_.get_pose(link_name[i]);

        for (int j = 0; j < sphere_num[i]; j++) {
            evaluat_sphere =
                start.translation +
                start.rotation.toRotationMatrix() * sphere_pos[i][j];
            double evaluat_dist = esdf_map_->GetPointDistance(evaluat_sphere);
            double adjusted_dist = evaluat_dist - sphere_r[i][j];
            std::cout << adjusted_dist << std::endl;

            marker.id += 1;
            marker.pose.position.x = evaluat_sphere[0];
            marker.pose.position.y = evaluat_sphere[1];
            marker.pose.position.z = evaluat_sphere[2];
            marker.scale.x = sphere_r[i][j];
            marker.scale.y = sphere_r[i][j];
            marker.scale.z = sphere_r[i][j];
            marker_array.markers.push_back(marker);
        }
    }

    sphere_pub_.publish(marker_array);
}

/**
 * @brief         用于创建当前 Cost 对象的副本。
 *
 * @return        cost_ptr
 * @attention
 */
cost_ptr Cost::clone() const { return std::make_shared<Cost>(*this); }

cost_t Cost::compute_cost(const observation_t& x, const input_t& u,
                          const reference_t& ref) {
    cost_t cost;

    // update model
    robot_model_.update_state(x);

    // target reaching cost
    if (local_track_mode) {
        Eigen::Vector3d error;
        error << x[0] - base_aim[0], x[1] - base_aim[1], 0;

        cost += error.transpose() * Q_linear_ * error;

        // preference for the manipulator angle
        // cost += last_u.segment(0, 2).norm() *
        //         (x.segment(3, 6) - pos_prefer).norm() * 20.0;
    } else {
        Eigen::Vector3d ref_t = ref.head<3>();
        Eigen::Quaterniond ref_q(ref.segment<4>(3));
        Eigen::Matrix<double, 6, 1> error;

        mppi::Pose current_pose = robot_model_.get_pose("panda_hand");
        mppi::Pose reference_pose(ref_t, ref_q);
        error = mppi::diff(current_pose, reference_pose);
        cost += error.head<3>().transpose() * Q_linear_ * error.head<3>();
        cost += error.tail<3>().transpose() * Q_angular_ * error.tail<3>();
    }

    // obstacle avoidance cost
    Eigen::Vector3d evaluat_sphere;
    Eigen::MatrixXd sphere_pos_mat(11, 3);
    mppi::Pose start;
    double min_dist = 100000.0;

    for (int i = 0; i < 7; i++) {
        mppi::Pose start = robot_model_.get_pose(link_name[i]);

        for (int j = 0; j < sphere_num[i]; j++) {
            evaluat_sphere =
                start.translation +
                start.rotation.toRotationMatrix() * sphere_pos[i][j];
            double evaluat_dist = esdf_map_->GetPointDistance(evaluat_sphere);
            double adjusted_dist = evaluat_dist - sphere_r[i][j];
            if (evaluat_dist != -10000 && adjusted_dist < min_dist)
                min_dist = adjusted_dist;

            sphere_pos_mat.row(sphere_row[i] + j) = evaluat_sphere.transpose();
        }
    }

    if (min_dist < safe_distance_) {
        cost += Q_obst_;
    } else {
        // self collision
        for (int i = 4; i < 7; i++) {
            for (int j = 0; j < sphere_num[i]; j++) {
                for (int k = 0; k < sphere_row[i]; k++) {
                    double evaluat_dist =
                        (sphere_pos_mat.row(sphere_row[i] + j) -
                         sphere_pos_mat.row(k))
                            .norm();
                    if (k == 0) evaluat_dist += -0.22;
                    double adjusted_dist = evaluat_dist - sphere_r[i][j] + 0.05;
                    if (adjusted_dist < min_dist) min_dist = adjusted_dist;
                }
            }
        }
        if (min_dist < safe_distance_) {
            cost += Q_obst_;
        } else {
            for (int k = sphere_row[4]; k < 11; k++) {
                if (sphere_pos_mat(k, 2) < 0.1) {
                    cost += Q_obst_;
                    break;
                }
            }
        }
    }

    if (cost < 0.001) cost += 10.0 * u.norm();

    return cost;
}

void Cost::set_reference_trajectory(const reference_trajectory_t& traj) {
    // TODO try direct copy
    timed_ref_.tt.clear();
    timed_ref_.tt.reserve(traj.tt.size());
    timed_ref_.rr.clear();
    for (unsigned int i = 0; i < traj.tt.size(); i++) {
        timed_ref_.rr.push_back(traj.rr[i]);
        timed_ref_.tt.push_back(traj.tt[i]);
    }
}

void Cost::interpolate_reference(const observation_t& /*x*/, reference_t& ref,
                                 const double t) {
    auto lower =
        std::lower_bound(timed_ref_.tt.begin(), timed_ref_.tt.end(), t);
    unsigned int offset = std::distance(timed_ref_.tt.begin(), lower);
    if (lower == timed_ref_.tt.end()) offset -= 1;
    ref = timed_ref_.rr[offset];
}

double Cost::get_stage_cost(const observation_t& x, const input_t& u,
                            const double t) {
    interpolate_reference(x, r_, t);
    return compute_cost(x, u, r_);
}

}  // namespace mppi