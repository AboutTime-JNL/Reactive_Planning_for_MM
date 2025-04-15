#include "parameters.h"

void fiesta::Parameters::SetParameters(const ros::NodeHandle &node) {
    node.param<double>("resolution", resolution_, 0.1);  // 网格距离
    node.param<int>("visualize_every_n_updates", visualize_every_n_updates_,
                    1);  // rviz可视化周期，以ESDF更新次数为单位

    /* slice visualization */
    double slice_vis_level_tmp;  //
    node.param<double>("slice_vis_max_dist", slice_vis_max_dist_,
                       2.0);  // ESDF显示的最大障碍物距离
    node.param<double>("slice_vis_level", slice_vis_level_tmp,
                       0.5);  // ESDF切片显示z轴位置
    slice_vis_level_ = (int)(slice_vis_level_tmp / resolution_);  //

    // ESDF和OGM的更新半径
    node.param<bool>(
        "global_update", global_update_,
        false);  // ESDF是否全局更新,ESDF的更新半径必须大于OGM的更新半径

    double radius_x, radius_y, radius_z;                      //
    node.param<double>("radius_x", radius_x, 1.f);            //
    node.param<double>("radius_y", radius_y, 1.f);            //
    node.param<double>("radius_z", radius_z, 0.7f);           //
    radius_ = Eigen::Vector3d(radius_x, radius_y, radius_z);  // ESDF更新半径

    // ESDF可视化参数
    node.param<bool>(
        "global_vis", global_vis_,
        true);  // ESDF是否全局显示，否的话同时受更新半径OGM显示限制
    node.param<double>("vis_radius_x", radius_x, 1.7f);           //
    node.param<double>("vis_radius_y", radius_y, 2.5f);           //
    node.param<double>("vis_radius_z", radius_z, 1.0f);           //
    vis_radius_ = Eigen::Vector3d(radius_x, radius_y, radius_z);  // 显示半径
    vis_center_ << 0.5, 0, 1.0;                                   // 显示中心

    double lx, ly, lz;
    double rx, ry, rz;

    // shelf2_desk
    node.param<double>("lx", lx, -1.5f);
    node.param<double>("ly", ly, -2.6f);
    node.param<double>("lz", lz, 0.f);
    node.param<double>("rx", rx, 1.5f);
    node.param<double>("ry", ry, 2.6f);
    node.param<double>("rz", rz, 1.6f);

    origin_ << lx, ly, lz;              // 地图左下角，grid编号为（0，0，0）
    l_cornor_ << lx, ly, lz;            //
    r_cornor_ << rx, ry, rz;            //
    map_size_ = r_cornor_ - l_cornor_;  // 地图大小
}