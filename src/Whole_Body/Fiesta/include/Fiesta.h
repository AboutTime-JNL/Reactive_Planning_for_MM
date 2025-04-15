#ifndef ESDF_TOOLS_INCLUDE_FIESTA_H_
#define ESDF_TOOLS_INCLUDE_FIESTA_H_
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include "ESDFMap.h"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <visualization_msgs/Marker.h>

namespace fiesta {

template <class DepthMsgType, class PoseMsgType>
class Fiesta {
   private:
    Parameters parameters_;
    ESDFMap *esdf_map_;
    bool new_msg_ = false;
    pcl::PointCloud<pcl::PointXYZ> cloud_;
    ros::Publisher slice_pub_, occupancy_pub_, text_pub_, vis_pub_;
    ros::Subscriber transform_sub_, depth_sub_;
    Eigen::Vector3d sync_pos_, cur_pos_;

    std::queue<std::tuple<ros::Time, Eigen::Vector3d>> transform_queue_;
    std::queue<DepthMsgType> depth_queue_;

    uint esdf_cnt_ = 0;

    void Visualization(ESDFMap *esdf_map, bool global_vis,
                       const std::string &text);

    void DepthCallback(const DepthMsgType &depth_map);

    void PoseCallback(const PoseMsgType &msg);

   public:
    Fiesta(ros::NodeHandle node);
    ~Fiesta();

    void Reset();

    double GetPointDistance(Eigen::Vector3d position);

    Eigen::Vector3d update_lower_bound, update_upper_bound;

    void UpdateEsdfEvent();
};

template <class DepthMsgType, class PoseMsgType>
Fiesta<DepthMsgType, PoseMsgType>::Fiesta(ros::NodeHandle node) {
    parameters_.SetParameters(node);
    esdf_map_ = new ESDFMap(parameters_.origin_, parameters_.resolution_,
                            parameters_.map_size_);

    update_lower_bound = Eigen::Vector3d::Zero();
    update_upper_bound = Eigen::Vector3d::Zero();

    depth_sub_ = node.subscribe("/camera/depth_registered/points", 1,
                                &Fiesta::DepthCallback, this);
    transform_sub_ = node.subscribe("/kinect/vrpn_client/estimated_transform",
                                    10, &Fiesta::PoseCallback, this);

    slice_pub_ =
        node.advertise<visualization_msgs::Marker>("ESDFMap/slice", 1, true);
    occupancy_pub_ =
        node.advertise<sensor_msgs::PointCloud2>("ESDFMap/occ_pc", 1, true);
    vis_pub_ =
        node.advertise<sensor_msgs::PointCloud2>("ESDFMap/occ_vis", 1, true);
    text_pub_ =
        node.advertise<visualization_msgs::Marker>("ESDFMap/text", 1, true);
}

template <class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::Reset() {
    esdf_map_->Reset();

    update_lower_bound = Eigen::Vector3d::Zero();
    update_upper_bound = Eigen::Vector3d::Zero();
}

template <class DepthMsgType, class PoseMsgType>
Fiesta<DepthMsgType, PoseMsgType>::~Fiesta() {
    delete esdf_map_;
}

template <class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::PoseCallback(const PoseMsgType &msg) {
    Eigen::Vector3d pos;

    pos = Eigen::Vector3d(msg->transform.translation.x,
                          msg->transform.translation.y,
                          msg->transform.translation.z);

    transform_queue_.push(std::make_tuple(msg->header.stamp, pos));
}

template <class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::DepthCallback(
    const DepthMsgType &depth_map) {
    depth_queue_.push(depth_map);
    ros::Time depth_time;
    double time_delay = 3e-3;
    while (!depth_queue_.empty()) {
        bool new_pos = false;
        depth_time = depth_queue_.front()->header.stamp;

        while (transform_queue_.size() > 1 &&
               std::get<0>(transform_queue_.front()) <=
                   depth_time + ros::Duration(time_delay)) {
            sync_pos_ = std::get<1>(transform_queue_.front());
            transform_queue_.pop();
            new_pos = true;
        }

        if (transform_queue_.empty() ||
            std::get<0>(transform_queue_.front()) <=
                depth_time + ros::Duration(time_delay)) {
            break;
        }

        if (!new_pos) {
            depth_queue_.pop();
            continue;
        }

        sensor_msgs::PointCloud2::ConstPtr tmp = depth_queue_.front();
        pcl::fromROSMsg(*tmp, cloud_);

        if ((int)cloud_.points.size() == 0) {
            depth_queue_.pop();
            continue;
        }

        new_msg_ = true;

        depth_queue_.pop();
        return;
    }
}

template <class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::UpdateEsdfEvent() {
    if (!new_msg_) return;
    new_msg_ = false;
    cur_pos_ = sync_pos_;

    if (parameters_.global_update_)
        esdf_map_->SetOriginalRange();
    else
        /** fix range **/
        // esdf_map_->SetUpdateRange(cur_pos_ - parameters_.radius_,
        //                           cur_pos_ + parameters_.radius_);

        /**dynamic range**/
        esdf_map_->SetUpdateRange(update_lower_bound, update_upper_bound);

    esdf_map_->SetAway();

    Eigen::Vector3i tmp_vox;
    Eigen::Vector3d tmp_pos;
    for (int i = 0; i < cloud_.size(); i++) {
        tmp_pos = Eigen::Vector3d(cloud_[i].x, cloud_[i].y, cloud_[i].z);
        esdf_map_->SetOccupancy(tmp_pos, 1);
    }
    esdf_map_->SetBack();

    if (esdf_map_->UpdateOccupancy()) {
        esdf_cnt_++;

        esdf_map_->UpdateESDF();
    }

    if (parameters_.visualize_every_n_updates_ != 0 &&
        esdf_cnt_ % parameters_.visualize_every_n_updates_ == 0) {
        Visualization(esdf_map_, parameters_.global_vis_, "");
    }
}

template <class DepthMsgType, class PoseMsgType>
void Fiesta<DepthMsgType, PoseMsgType>::Visualization(ESDFMap *esdf_map,
                                                      bool global_vis,
                                                      const std::string &text) {
    if (esdf_map != nullptr) {
        esdf_map->SetOriginalRange();

        sensor_msgs::PointCloud pc;
        esdf_map->GetPointCloud(pc);
        sensor_msgs::PointCloud2 pc2;
        convertPointCloudToPointCloud2(pc, pc2);
        pc2.header.stamp = pc.header.stamp;
        pc2.header.frame_id = "world";
        occupancy_pub_.publish(pc2);

        // if (global_vis)
        //     esdf_map->SetOriginalRange();
        // else {
        //     // Eigen::Vector3d upper_bound = cur_pos_ +
        //     // parameters_.vis_radius_;
        //     // upper_bound(1) = cur_pos_(1) + 1.0;
        //     // esdf_map->SetUpdateRange(cur_pos_ - parameters_.vis_radius_,
        //     // upper_bound, false);
        //     // esdf_map->SetUpdateRange(
        //     //     parameters_.vis_center_ - parameters_.vis_radius_,
        //     //     parameters_.vis_center_ + parameters_.vis_radius_, false);
        //     esdf_map_->SetUpdateRange(cur_pos_ - parameters_.vis_radius_,
        //                               cur_pos_ + parameters_.vis_radius_,
        //                               false);
        // }

        // esdf_map->GetPointCloud(pc);
        // convertPointCloudToPointCloud2(pc, pc2);
        // pc2.header.stamp = pc.header.stamp;
        // pc2.header.frame_id = "world";
        // vis_pub_.publish(pc2);

        // visualization_msgs::Marker slice_marker;
        // esdf_map->GetSliceMarker(slice_marker, parameters_.slice_vis_level_,
        //                          100, Eigen::Vector4d(0, 1.0, 0, 1),
        //                          parameters_.slice_vis_max_dist_);
        // slice_pub_.publish(slice_marker);
    }
}

template <class DepthMsgType, class PoseMsgType>
double Fiesta<DepthMsgType, PoseMsgType>::GetPointDistance(
    Eigen::Vector3d position) {
    return esdf_map_->GetDistance(position);
}

}  // namespace fiesta
#endif  // ESDF_TOOLS_INCLUDE_FIESTA_H_
