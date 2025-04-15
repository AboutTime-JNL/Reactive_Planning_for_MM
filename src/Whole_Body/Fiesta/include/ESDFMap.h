#ifndef ESDF_MAP_H
#define ESDF_MAP_H

#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/Marker.h>
#include <iostream>
#include <queue>
#include <vector>
#include "parameters.h"

namespace fiesta {
class ESDFMap {
    // Type of queue element to be used in priority queue
    struct QueueElement {
        Eigen::Vector3i point_;
        double distance_;
        bool operator<(const QueueElement &element) const {
            return distance_ > element.distance_;
        }
    };

   private:
    // parameters & method for occupancy information updating
    bool Exist(const int &idx);
    double Dist(Eigen::Vector3i a, Eigen::Vector3i b);

    // parameters & methods for conversion between Pos, Vox & Idx
    bool PosInMap(Eigen::Vector3d pos);
    bool VoxInRange(Eigen::Vector3i vox, bool current_vec = true);
    void Vox2Pos(Eigen::Vector3i vox, Eigen::Vector3d &pos);
    int Vox2Idx(Eigen::Vector3i vox);
    int Vox2Idx(Eigen::Vector3i vox, int sub_sampling_factor);
    void Pos2Vox(Eigen::Vector3d pos, Eigen::Vector3i &vox);
    Eigen::Vector3i Idx2Vox(int idx);

    // data are saved in vector
    std::vector<unsigned char> occupancy_buffer_;  // 0 is free, 1 is occupied
    std::vector<double> distance_buffer_;
    std::vector<Eigen::Vector3i> closest_obstacle_;
    std::vector<int> head_, prev_, next_;

   public:
    std::queue<QueueElement> insert_queue_;
    std::queue<QueueElement> delete_queue_;
    std::queue<QueueElement> update_queue_;

   private:
    // Map Properties
    Eigen::Vector3d origin_;
    int reserved_idx_4_undefined_;
    int total_time_ = 0;
    int infinity_, undefined_;
    double resolution_, resolution_inv_;
    Eigen::Vector3i max_vec_, min_vec_, last_max_vec_, last_min_vec_;

    Eigen::Vector3d map_size_;
    Eigen::Vector3d min_range_, max_range_;  // map range in pos
    Eigen::Vector3i grid_size_;              // map range in index
    int grid_size_yz_;

    // DLL Operations
    void DeleteFromList(int link, int idx);
    void InsertIntoList(int link, int idx);

   public:
    int grid_total_size_;
    ESDFMap(Eigen::Vector3d origin, double resolution,
            Eigen::Vector3d map_size);

    ~ESDFMap() {
        // TODO: implement this
    }

    // Reset
    void Reset();

    bool UpdateOccupancy();
    void UpdateESDF();

    // Occupancy Management
    void SetOccupancy(Eigen::Vector3d pos, int occ);
    void SetOccupancy(Eigen::Vector3i vox, int occ);

    // Distance Field Management
    double GetDistance(Eigen::Vector3d pos);
    double GetDistance(Eigen::Vector3i vox);

    // Visualization
    void GetPointCloud(sensor_msgs::PointCloud &m);
    void GetSliceMarker(visualization_msgs::Marker &m, int slice, int id,
                        Eigen::Vector4d color, double max_dist);

    // Local Range
    void SetUpdateRange(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos,
                        bool new_vec = true);
    void SetOriginalRange();

    // For Deterministic Occupancy Grid
    void SetAway();
    void SetAway(Eigen::Vector3i left, Eigen::Vector3i right);
    void SetBack();
    void SetBack(Eigen::Vector3i left, Eigen::Vector3i right);
};
}  // namespace fiesta

#endif  // ESDF_MAP_H
