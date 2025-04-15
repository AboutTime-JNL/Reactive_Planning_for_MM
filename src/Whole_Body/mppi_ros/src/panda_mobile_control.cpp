#include "controller_interface.h"
#include "core/dynamics.h"

#include <ros/ros.h>
#include <chrono>

int main(int argc, char** argv) {
    // ros interface
    ros::init(argc, argv, "panda_mobile_control_node");

    ros::NodeHandle nh("~");
    ros::NodeHandle nh_sensor("~");
    ros::NodeHandle nh_local("~");

    /* init the joint pose publisher */
    ros::Publisher state_publisher =
        nh.advertise<sensor_msgs::JointState>("/sim_joint_states", 10);
    sensor_msgs::JointState joint_state;
    joint_state.name = {"x_base_joint",
                        "y_base_joint",
                        "w_base_joint",
                        "jaka_shoulder_pan_joint",
                        "jaka_shoulder_lift_joint",
                        "jaka_elbow_joint",
                        "jaka_wrist_1_joint",
                        "jaka_wrist_2_joint",
                        "jaka_wrist_3_joint"};
    joint_state.position.resize(9);
    joint_state.header.frame_id = "world";

    // controller
    auto controller = mppi::Controller(nh, nh_sensor, nh_local);

    // simulation
    auto simulation = mppi::Dynamics();
    mppi::input_t u = simulation.get_zero_input();
    double sim_dt = controller.config_.sim_dt;

    // Wait for the game information
    while (ros::ok() && !controller.game_start_) {
        ros::spinOnce();
        ros::Duration(0.1).sleep();
    }

    // Wait for the initial joint state
    simulation.reset(controller.start_joint_state_, 0.0);
    ROS_INFO_STREAM("Position Sensors init finish");

    controller.set_observation(simulation.get_state(), simulation.get_time());

    controller.start();

    ros::Duration(1).sleep();

    while (ros::ok()) {
        auto start = std::chrono::steady_clock::now();

        ros::spinOnce();

        // get the joint state
        if (controller.game_reset_) {
            simulation.reset(controller.start_joint_state_,
                             simulation.get_time());
            controller.set_observation(simulation.get_state(),
                                       simulation.get_time());
            controller.game_reset_ = false;
            ros::Duration(1).sleep();
        } else
            simulation.step(u, sim_dt);

        for (unsigned int i = 0; i < mppi::Dim::STATE_DIMENSION; i++)
            joint_state.position[i] = simulation.get_state()[i];
        joint_state.header.stamp = ros::Time::now();
        state_publisher.publish(joint_state);

        controller.set_observation(simulation.get_state(),
                                   simulation.get_time());
        controller.get_input(simulation.get_state(), u, simulation.get_time());

        auto end = std::chrono::steady_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count() /
            1000.0;
        if (sim_dt - elapsed > 0) ros::Duration(sim_dt - elapsed).sleep();
    }

    return 0;
}
