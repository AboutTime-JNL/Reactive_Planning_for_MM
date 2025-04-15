cmake_minimum_required(VERSION 3.10)    # CMake版本
project(fiesta)                         # 软件包名

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        visualization_msgs
        cv_bridge
        )                               # 查找编译依赖的Catkin包（声明依赖库）,REQUIRED:是否强制要求该包，COMPONENTS:是否强制要求该包中某个组件
                                        # 将catkin包作为catkin的组件，可以直接将include纳入到catkin_INCLUDE_DIRS中

find_package(Eigen3 REQUIRED)           # 查找编译依赖的非Catkin包（声明依赖库）
find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS visualization_msgs cv_bridge
  DEPENDS Eigen3 PCL OpenCV
)                                       # 传递给其他依赖功能包的信息:INCLUDE_DIRS-本功能包对外头文件位置;LIBRARIES-本功能包对外库文件位置;
                                        # CATKIN_DEPENDS-本功能包运行需要依赖的catkin包;DEPENDS-本功能包运行需要依赖的包

include_directories(
    include
    SYSTEM
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)                                       # 添加依赖的头文件位置

set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -O3 -Wall -pthread")   
                                        # -Wextra -Werror
set(CMAKE_BUILD_TYPE "RELEASE")

add_library(${PROJECT_NAME}
        include/parameters.h
        src/parameters.cpp
        include/ESDFMap.h
        src/ESDFMap.cpp
        include/Fiesta.h)#生成库文件

add_dependencies(${PROJECT_NAME} 
        ${${PROJECT_NAME}_EXPORTED_TARGETS} 
        ${catkin_EXPORTED_TARGETS})     # 确定在生成库文件前需要先生成的其他依赖库，如message，service，
                                        # ${PROJECT_NAME}_EXPORTED_TARGETS：指本功能包中定义的message，service等
        
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        )                               # 确定库文件依赖的其他库文件

