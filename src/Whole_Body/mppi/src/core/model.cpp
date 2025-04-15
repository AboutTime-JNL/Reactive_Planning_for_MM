#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "core/model.h"

using namespace pinocchio;

namespace mppi {

Eigen::Matrix<double, 6, 1> diff(const Pose& p1, const Pose& p2) {
    return log6(SE3(p1.rotation, p1.translation)
                    .actInv(SE3(p2.rotation, p2.translation)))
        .toVector();
}

Pose operator*(const Pose& p1, const Pose& p2) {
    Pose res;
    SE3 temp(
        SE3(p1.rotation, p1.translation).act(SE3(p2.rotation, p2.translation)));
    res.translation = temp.translation();
    res.rotation = temp.rotation();
    return res;
}

RobotModel::~RobotModel() {
    delete model_;
    delete data_;
}

// deep copy
RobotModel::RobotModel(const RobotModel& rhs) {
    model_ = new pinocchio::Model(*rhs.model_);
    data_ = new pinocchio::Data(*rhs.data_);
}

bool RobotModel::init_from_xml(const std::string& robot_description) {
    try {
        model_ = new Model();
        pinocchio::urdf::buildModelFromXML(robot_description, *model_);
        data_ = new Data(*model_);
    } catch (std::runtime_error& exc) {
        std::cout << exc.what();
        return false;
    } catch (...) {
        std::cout << "Unknown exception caught while building model."
                  << std::endl;
        return false;
    }
    return true;
}

void RobotModel::update_state(const Eigen::VectorXd& q) {
    forwardKinematics(*model_, *data_, q);
    updateFramePlacements(*model_, *data_);
}

Pose RobotModel::get_pose(const std::string& frame) const {
    Pose pose;
    pose.translation = data_->oMf[model_->getFrameId(frame)].translation();
    pose.rotation = data_->oMf[model_->getFrameId(frame)].rotation();
    return pose;
}

}  // namespace mppi