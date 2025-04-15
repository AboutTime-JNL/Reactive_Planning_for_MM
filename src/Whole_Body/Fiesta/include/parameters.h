#ifndef ESDF_TOOLS_INCLUDE_PARAMETERS_H_
#define ESDF_TOOLS_INCLUDE_PARAMETERS_H_
#include <ros/ros.h>
#include <Eigen/Eigen>
#define BLOCK
#define BITWISE
#define DEBUG

namespace fiesta {

// Connectivity used in BFS
// region DIRECTION
const static int num_dirs_ = 24;  // faces 2 steps
const Eigen::Vector3i dirs_[num_dirs_] = {
    Eigen::Vector3i(-1, 0, 0),  Eigen::Vector3i(1, 0, 0),
    Eigen::Vector3i(0, -1, 0),  Eigen::Vector3i(0, 1, 0),
    Eigen::Vector3i(0, 0, -1),  Eigen::Vector3i(0, 0, 1),

    Eigen::Vector3i(-1, -1, 0), Eigen::Vector3i(1, 1, 0),
    Eigen::Vector3i(0, -1, -1), Eigen::Vector3i(0, 1, 1),
    Eigen::Vector3i(-1, 0, -1), Eigen::Vector3i(1, 0, 1),
    Eigen::Vector3i(-1, 1, 0),  Eigen::Vector3i(1, -1, 0),
    Eigen::Vector3i(0, -1, 1),  Eigen::Vector3i(0, 1, -1),
    Eigen::Vector3i(1, 0, -1),  Eigen::Vector3i(-1, 0, 1),

    Eigen::Vector3i(-2, 0, 0),  Eigen::Vector3i(2, 0, 0),
    Eigen::Vector3i(0, -2, 0),  Eigen::Vector3i(0, 2, 0),
    Eigen::Vector3i(0, 0, -2),  Eigen::Vector3i(0, 0, 2)};
// endregion

struct Parameters {
    // resolution
    double resolution_;
    // array implementation only
    Eigen::Vector3d l_cornor_, r_cornor_, map_size_;
    // visualization
    double slice_vis_max_dist_;
    int slice_vis_level_;
    // frequency of visualization
    int visualize_every_n_updates_;
    // local map
    bool global_vis_, global_update_;
    Eigen::Vector3d radius_, vis_radius_, vis_center_;
    // transforms
    Eigen::Vector3d origin_;

    void SetParameters(const ros::NodeHandle &node);
};
}  // namespace fiesta
#endif  // ESDF_TOOLS_INCLUDE_PARAMETERS_H_
