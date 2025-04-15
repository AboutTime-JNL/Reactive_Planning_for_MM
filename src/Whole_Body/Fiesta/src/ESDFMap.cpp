//
// Created by tommy on 12/17/18.
//

#include "ESDFMap.h"
#include <math.h>
#include <time.h>

using std::cout;
using std::endl;

bool fiesta::ESDFMap::Exist(const int &idx) {
    return occupancy_buffer_[idx] == 1;
}

void fiesta::ESDFMap::DeleteFromList(int link, int idx) {
    if (prev_[idx] != undefined_)
        next_[prev_[idx]] = next_[idx];
    else
        head_[link] = next_[idx];
    if (next_[idx] != undefined_) prev_[next_[idx]] = prev_[idx];
    prev_[idx] = next_[idx] = undefined_;
}

void fiesta::ESDFMap::InsertIntoList(int link, int idx) {
    if (head_[link] == undefined_)
        head_[link] = idx;
    else {
        prev_[head_[link]] = idx;
        next_[idx] = head_[link];
        head_[link] = idx;
    }
}

// region CONVERSION

bool fiesta::ESDFMap::PosInMap(Eigen::Vector3d pos) {
    if (pos(0) < min_range_(0) || pos(1) < min_range_(1) ||
        pos(2) < min_range_(2)) {
        return false;
    }

    if (pos(0) > max_range_(0) || pos(1) > max_range_(1) ||
        pos(2) > max_range_(2)) {
        return false;
    }
    return true;
}

bool fiesta::ESDFMap::VoxInRange(Eigen::Vector3i vox, bool current_vec) {
    if (current_vec)
        return (vox(0) >= min_vec_(0) && vox(0) <= max_vec_(0) &&
                vox(1) >= min_vec_(1) && vox(1) <= max_vec_(1) &&
                vox(2) >= min_vec_(2) && vox(2) <= max_vec_(2));
    else
        return (vox(0) >= last_min_vec_(0) && vox(0) <= last_max_vec_(0) &&
                vox(1) >= last_min_vec_(1) && vox(1) <= last_max_vec_(1) &&
                vox(2) >= last_min_vec_(2) && vox(2) <= last_max_vec_(2));
}

void fiesta::ESDFMap::Pos2Vox(Eigen::Vector3d pos, Eigen::Vector3i &vox) {
    for (int i = 0; i < 3; ++i)
        vox(i) = floor((pos(i) - origin_(i)) / resolution_);
}

void fiesta::ESDFMap::Vox2Pos(Eigen::Vector3i vox, Eigen::Vector3d &pos) {
    for (int i = 0; i < 3; ++i)
        pos(i) = (vox(i) + 0.5) * resolution_ + origin_(i);
}

int fiesta::ESDFMap::Vox2Idx(Eigen::Vector3i vox) {
    if (vox(0) == undefined_) return reserved_idx_4_undefined_;
    return vox(0) * grid_size_yz_ + vox(1) * grid_size_(2) + vox(2);
}

int fiesta::ESDFMap::Vox2Idx(Eigen::Vector3i vox, int sub_sampling_factor) {
    if (vox(0) == undefined_) return reserved_idx_4_undefined_;
    return vox(0) * grid_size_yz_ / sub_sampling_factor / sub_sampling_factor /
               sub_sampling_factor +
           vox(1) * grid_size_(2) / sub_sampling_factor / sub_sampling_factor +
           vox(2) / sub_sampling_factor;
}

Eigen::Vector3i fiesta::ESDFMap::Idx2Vox(int idx) {
    return Eigen::Vector3i(idx / grid_size_yz_,
                           idx % (grid_size_yz_) / grid_size_(2),
                           idx % grid_size_(2));
}

// endregion

double fiesta::ESDFMap::Dist(Eigen::Vector3i a, Eigen::Vector3i b) {
    return (b - a).cast<double>().norm() * resolution_;
    //        return (b - a).squaredNorm();
    // TODO: may use square root & * resolution_ at last together to speed up
}

fiesta::ESDFMap::ESDFMap(Eigen::Vector3d origin, double resolution,
                         Eigen::Vector3d map_size)
    : origin_(origin), resolution_(resolution), map_size_(map_size) {
    resolution_inv_ = 1 / resolution;

    for (int i = 0; i < 3; ++i) grid_size_(i) = ceil(map_size(i) / resolution);

    min_range_ = origin;
    max_range_ = origin + map_size;
    grid_size_yz_ = grid_size_(1) * grid_size_(2);
    infinity_ = 10000;
    undefined_ = -10000;

    grid_total_size_ = grid_size_(0) * grid_size_yz_;
    reserved_idx_4_undefined_ = grid_total_size_;

    SetOriginalRange();

    occupancy_buffer_.resize(grid_total_size_);

    distance_buffer_.resize(grid_total_size_);

    closest_obstacle_.resize(grid_total_size_);

    std::fill(distance_buffer_.begin(), distance_buffer_.end(),
              (double)undefined_);

    std::fill(occupancy_buffer_.begin(), occupancy_buffer_.end(), 0);
    std::fill(closest_obstacle_.begin(), closest_obstacle_.end(),
              Eigen::Vector3i(undefined_, undefined_, undefined_));

    head_.resize(grid_total_size_ + 1);
    prev_.resize(grid_total_size_);
    next_.resize(grid_total_size_);
    std::fill(head_.begin(), head_.end(), undefined_);
    std::fill(prev_.begin(), prev_.end(), undefined_);
    std::fill(next_.begin(), next_.end(), undefined_);
}

void fiesta::ESDFMap::Reset() {
    SetOriginalRange();

    std::fill(distance_buffer_.begin(), distance_buffer_.end(),
              (double)undefined_);
    std::fill(occupancy_buffer_.begin(), occupancy_buffer_.end(), 0);
    std::fill(closest_obstacle_.begin(), closest_obstacle_.end(),
              Eigen::Vector3i(undefined_, undefined_, undefined_));

    std::fill(head_.begin(), head_.end(), undefined_);
    std::fill(prev_.begin(), prev_.end(), undefined_);
    std::fill(next_.begin(), next_.end(), undefined_);
}

bool fiesta::ESDFMap::UpdateOccupancy() {
    return !insert_queue_.empty() || !delete_queue_.empty();
}

void fiesta::ESDFMap::UpdateESDF() {
    // std::cout << "Delete: " << delete_queue_.size()
    //           << " Insert: " << insert_queue_.size() << std::endl;
    while (!insert_queue_.empty()) {
        QueueElement xx = insert_queue_.front();
        insert_queue_.pop();
        int idx = Vox2Idx(xx.point_);
        if (Exist(idx)) {
            // Exist after a whole brunch of updates
            // delete previous link & create a new linked-list
            DeleteFromList(Vox2Idx(closest_obstacle_[idx]), idx);
            closest_obstacle_[idx] = xx.point_;

            distance_buffer_[idx] = 0.0;

            InsertIntoList(idx, idx);
            update_queue_.push(xx);
        }
    }
    while (!delete_queue_.empty()) {
        QueueElement xx = delete_queue_.front();

        delete_queue_.pop();
        int idx = Vox2Idx(xx.point_);
        if (!Exist(idx)) {
            // doesn't Exist after a whole brunch of updates

            int next_obs_idx;
            for (int obs_idx = head_[idx]; obs_idx != undefined_;
                 obs_idx = next_obs_idx) {
                closest_obstacle_[obs_idx] =
                    Eigen::Vector3i(undefined_, undefined_, undefined_);
                Eigen::Vector3i obs_vox = Idx2Vox(obs_idx);

                double distance = infinity_;
                // find neighborhood whose closest obstacles Exist
                for (const auto &dir : dirs_) {
                    Eigen::Vector3i new_pos = obs_vox + dir;
                    int new_pos_idx = Vox2Idx(new_pos);
                    if (VoxInRange(new_pos) &&
                        closest_obstacle_[new_pos_idx](0) != undefined_ &&
                        Exist(Vox2Idx(closest_obstacle_[new_pos_idx]))) {
                        // if in range and closest obstacles Exist
                        double tmp =
                            Dist(obs_vox, closest_obstacle_[new_pos_idx]);
                        if (tmp < distance) {
                            distance = tmp;
                            closest_obstacle_[obs_idx] =
                                closest_obstacle_[new_pos_idx];
                        }
                        break;
                    }  // if
                }  // for neighborhood

                // destroy the linked-list
                prev_[obs_idx] = undefined_;
                next_obs_idx = next_[obs_idx];
                next_[obs_idx] = undefined_;

                distance_buffer_[obs_idx] = distance;

                if (distance < infinity_) {
                    update_queue_.push(QueueElement{obs_vox, distance});
                }
                int new_obs_idx = Vox2Idx(closest_obstacle_[obs_idx]);
                InsertIntoList(new_obs_idx, obs_idx);
            }  // for obs_idx
            head_[idx] = undefined_;
        }  // if
    }  // delete_queue_
    int times = 0, change_num = 0;

    while (!update_queue_.empty()) {
        QueueElement xx = update_queue_.front();

        update_queue_.pop();
        int idx = Vox2Idx(xx.point_);
        if (xx.distance_ != distance_buffer_[idx]) continue;
        times++;
        bool change = false;
        for (int i = 0; i < num_dirs_; i++) {
            Eigen::Vector3i new_pos = xx.point_ + dirs_[i];
            if (VoxInRange(new_pos)) {
                int new_pos_idx = Vox2Idx(new_pos);
                if (closest_obstacle_[new_pos_idx](0) == undefined_) continue;
                double tmp = Dist(xx.point_, closest_obstacle_[new_pos_idx]);

                if (distance_buffer_[idx] > tmp ||
                    distance_buffer_[idx] == undefined_) {
                    distance_buffer_[idx] = tmp;

                    change = true;
                    DeleteFromList(Vox2Idx(closest_obstacle_[idx]), idx);

                    int new_obs_idx = Vox2Idx(closest_obstacle_[new_pos_idx]);
                    InsertIntoList(new_obs_idx, idx);
                    closest_obstacle_[idx] = closest_obstacle_[new_pos_idx];
                }
            }
        }

        if (change) {
            change_num++;
            update_queue_.push(QueueElement{xx.point_, distance_buffer_[idx]});
            continue;
        }

        int new_obs_idx = Vox2Idx(closest_obstacle_[idx]);
        for (const auto &dir : dirs_) {
            Eigen::Vector3i new_pos = xx.point_ + dir;
            if (VoxInRange(new_pos)) {
                int new_pos_id = Vox2Idx(new_pos);

                double tmp = Dist(new_pos, closest_obstacle_[idx]);
                if (distance_buffer_[new_pos_id] > tmp ||
                    distance_buffer_[new_pos_id] == undefined_) {
                    distance_buffer_[new_pos_id] = tmp;

                    DeleteFromList(Vox2Idx(closest_obstacle_[new_pos_id]),
                                   new_pos_id);

                    InsertIntoList(new_obs_idx, new_pos_id);
                    closest_obstacle_[new_pos_id] = closest_obstacle_[idx];
                    update_queue_.push(QueueElement{new_pos, tmp});
                }
            }
        }
    }
    total_time_ += times;
}

void fiesta::ESDFMap::SetOccupancy(Eigen::Vector3d pos, int occ) {
    if (occ != 1 && occ != 0) {
        cout << "occ value error!" << occ << endl;
        return;
    }

    if (!PosInMap(pos)) {
        return;
    }

    Eigen::Vector3i vox;
    Pos2Vox(pos, vox);
    SetOccupancy(vox, occ);
}

void fiesta::ESDFMap::SetOccupancy(Eigen::Vector3i vox, int occ) {
    int idx = Vox2Idx(vox);

    if (!VoxInRange(vox)) {
        return;
    }

    if (occupancy_buffer_[idx] != occ && occupancy_buffer_[idx] != (occ | 2)) {
        if (occ == 1)
            insert_queue_.push(QueueElement{vox, 0.0});
        else
            delete_queue_.push(QueueElement{vox, (double)infinity_});
    }
    occupancy_buffer_[idx] = occ;
    if (distance_buffer_[idx] < 0) {
        distance_buffer_[idx] = infinity_;
        InsertIntoList(reserved_idx_4_undefined_, idx);
    }
}

double fiesta::ESDFMap::GetDistance(Eigen::Vector3d pos) {
    if (!PosInMap(pos)) return undefined_;

    Eigen::Vector3i vox;
    Pos2Vox(pos, vox);

    return GetDistance(vox);
}

double fiesta::ESDFMap::GetDistance(Eigen::Vector3i vox) {
    return distance_buffer_[Vox2Idx(vox)] < 0 ? infinity_
                                              : distance_buffer_[Vox2Idx(vox)];
}
// region VISUALIZATION

void fiesta::ESDFMap::GetPointCloud(sensor_msgs::PointCloud &m) {
    m.header.frame_id = "world";
    m.points.clear();
    for (int x = min_vec_(0); x <= max_vec_(0); ++x)
        for (int y = min_vec_(1); y <= max_vec_(1); ++y)
            for (int z = min_vec_(2); z <= max_vec_(2); ++z) {
                if (!Exist(Vox2Idx(Eigen::Vector3i(x, y, z)))) continue;

                Eigen::Vector3d pos;
                Vox2Pos(Eigen::Vector3i(x, y, z), pos);

                geometry_msgs::Point32 p;
                p.x = pos(0);
                p.y = pos(1);
                p.z = pos(2);
                m.points.push_back(p);
            }
}

inline std_msgs::ColorRGBA RainbowColorMap(double h) {
    std_msgs::ColorRGBA color;
    color.a = 1;
    // blend over HSV-values (more colors)
    if (h < 0) {
        color.r = 1.0;
        color.g = 1.0;
        color.b = 1.0;
        return color;
    }

    double s = 1.0;
    double v = 1.0;

    h -= floor(h);
    h *= 6;
    int i;
    double m, n, f;

    i = floor(h);
    f = h - i;
    if (!(i & 1)) f = 1 - f;  // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);

    switch (i) {
        case 6:
        case 0:
            color.r = v;
            color.g = n;
            color.b = m;
            break;
        case 1:
            color.r = n;
            color.g = v;
            color.b = m;
            break;
        case 2:
            color.r = m;
            color.g = v;
            color.b = n;
            break;
        case 3:
            color.r = m;
            color.g = n;
            color.b = v;
            break;
        case 4:
            color.r = n;
            color.g = m;
            color.b = v;
            break;
        case 5:
            color.r = v;
            color.g = m;
            color.b = n;
            break;
        default:
            color.r = 1;
            color.g = 0.5;
            color.b = 0.5;
            break;
    }

    return color;
}

void fiesta::ESDFMap::GetSliceMarker(visualization_msgs::Marker &m, int slice,
                                     int id, Eigen::Vector4d color,
                                     double max_dist) {
    m.header.frame_id = "world";
    m.id = id;
    m.type = visualization_msgs::Marker::POINTS;
    m.action = visualization_msgs::Marker::MODIFY;
    m.scale.x = resolution_;
    m.scale.y = resolution_;
    m.scale.z = resolution_;
    m.pose.orientation.w = 1;
    m.pose.orientation.x = 0;
    m.pose.orientation.y = 0;
    m.pose.orientation.z = 0;

    m.points.clear();
    m.colors.clear();
    // iterate the map
    std_msgs::ColorRGBA c;
    for (int x = min_vec_(0); x <= max_vec_(0); ++x)
        for (int y = min_vec_(1); y <= max_vec_(1); ++y) {
            int z = slice;
            Eigen::Vector3i vox = Eigen::Vector3i(x, y, z);
            if (distance_buffer_[Vox2Idx(vox)] < 0 ||
                distance_buffer_[Vox2Idx(vox)] >= infinity_)
                continue;

            Eigen::Vector3d pos;
            Vox2Pos(vox, pos);

            geometry_msgs::Point p;
            p.x = pos(0);
            p.y = pos(1);
            p.z = pos(2);

            c = RainbowColorMap(distance_buffer_[Vox2Idx(vox)] <= max_dist
                                    ? distance_buffer_[Vox2Idx(vox)] / max_dist
                                    : -1);
            if (distance_buffer_[Vox2Idx(vox)] <= max_dist) {
                m.points.push_back(p);
                m.colors.push_back(c);
            }
        }
}

// endregion
// region LOCAL vs GLOBAL

void fiesta::ESDFMap::SetUpdateRange(Eigen::Vector3d min_pos,
                                     Eigen::Vector3d max_pos, bool new_vec) {
    min_pos(0) = std::max(min_pos(0), min_range_(0));
    min_pos(1) = std::max(min_pos(1), min_range_(1));
    min_pos(2) = std::max(min_pos(2), min_range_(2));

    max_pos(0) = std::min(max_pos(0), max_range_(0));
    max_pos(1) = std::min(max_pos(1), max_range_(1));
    max_pos(2) = std::min(max_pos(2), max_range_(2));
    if (new_vec) {
        last_min_vec_ = min_vec_;
        last_max_vec_ = max_vec_;
    }
    Pos2Vox(min_pos, min_vec_);
    Pos2Vox(max_pos - Eigen::Vector3d(resolution_ / 2, resolution_ / 2,
                                      resolution_ / 2),
            max_vec_);

    min_vec_(0) = std::max(min_vec_(0), 0);
    min_vec_(1) = std::max(min_vec_(1), 0);
    min_vec_(2) = std::max(min_vec_(2), 0);

    max_vec_(0) = std::min(max_vec_(0), grid_size_(0) - 1);
    max_vec_(1) = std::min(max_vec_(1), grid_size_(1) - 1);
    max_vec_(2) = std::min(max_vec_(2), grid_size_(2) - 1);
}

void fiesta::ESDFMap::SetOriginalRange() {
    min_vec_ << 0, 0, 0;
    max_vec_ << grid_size_(0) - 1, grid_size_(1) - 1, grid_size_(2) - 1;
    last_min_vec_ = min_vec_;
    last_max_vec_ = max_vec_;
}

// endregion
void fiesta::ESDFMap::SetAway() { SetAway(min_vec_, max_vec_); }
void fiesta::ESDFMap::SetAway(Eigen::Vector3i left, Eigen::Vector3i right) {
    for (int i = left(0); i <= right(0); i++)
        for (int j = left(1); j <= right(1); j++)
            for (int k = left(2); k <= right(2); k++)
                occupancy_buffer_[Vox2Idx(Eigen::Vector3i(i, j, k))] |= 2;
}

void fiesta::ESDFMap::SetBack() { SetBack(min_vec_, max_vec_); }
void fiesta::ESDFMap::SetBack(Eigen::Vector3i left, Eigen::Vector3i right) {
    for (int i = left(0); i <= right(0); i++)
        for (int j = left(1); j <= right(1); j++)
            for (int k = left(2); k <= right(2); k++)
                if (occupancy_buffer_[Vox2Idx(Eigen::Vector3i(i, j, k))] >= 2)
                    SetOccupancy(Eigen::Vector3i(i, j, k), 0);
}