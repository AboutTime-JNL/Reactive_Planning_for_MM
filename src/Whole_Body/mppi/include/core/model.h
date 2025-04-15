#pragma once
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/fwd.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

namespace mppi {

struct Pose {
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;

    Pose() = default;
    Pose(Eigen::Vector3d t, Eigen::Quaterniond r)
        : translation(t), rotation(r) {};
};

mppi::Pose operator*(const mppi::Pose&, const mppi::Pose&);
Eigen::Matrix<double, 6, 1> diff(const mppi::Pose&, const mppi::Pose&);

class RobotModel {
   public:
    using Vector6d = Eigen::Matrix<double, 6, 1>;

    RobotModel() = default;
    ~RobotModel();

    RobotModel(const RobotModel& rhs);
    /**
     *
     * @param robot_description
     * @return
     */
    bool init_from_xml(const std::string& robot_description);

    /**
     *
     * @param q
     */
    void update_state(const Eigen::VectorXd& q);

    /**
     *
     * @param frame
     */
    mppi::Pose get_pose(const std::string& frame) const;

   private:
    pinocchio::Model* model_;
    pinocchio::Data* data_;
};

}  // namespace mppi
