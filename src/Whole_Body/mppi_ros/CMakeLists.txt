cmake_minimum_required(VERSION 3.10)
project(mppi_ros)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED 3.3)

set(PACKAGE_DEPENDENCIES
        mppi
        sensor_msgs
        std_msgs
        roscpp
        fiesta
        roslib
        tf2
        tf2_ros
        )

find_package(catkin REQUIRED COMPONENTS ${PACKAGE_DEPENDENCIES})


catkin_package(
        LIBRARIES ${PROJECT_NAME}
        INCLUDE_DIRS include
        CATKIN_DEPENDS ${PACKAGE_DEPENDENCIES})

set(COMMON_LIBRARIES ${catkin_LIBRARIES})
set(SYSTEM_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

include_directories(include SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Libraries
add_library(mppi_threading src/threading/Rate.cpp src/threading/Worker.cpp src/threading/WorkerManager.cpp)

add_library(${PROJECT_NAME}
        src/controller_interface.cpp
        src/ros_params.cpp )
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} mppi_threading)
target_link_libraries(${PROJECT_NAME} ${COMMON_LIBRARIES} mppi_threading)

add_executable(panda_mobile_control src/panda_mobile_control.cpp)
target_link_libraries(panda_mobile_control ${PROJECT_NAME} ${catkin_LIBRARIES})
add_dependencies(panda_mobile_control ${PROJECT_NAME})

install(DIRECTORY launch resources
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )