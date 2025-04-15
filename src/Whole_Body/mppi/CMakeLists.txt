cmake_minimum_required(VERSION 3.10)
project(mppi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(Eigen3 REQUIRED 3.3)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(pinocchio REQUIRED)

set(PACKAGE_DEPENDENCIES
        fiesta
        )

find_package(catkin REQUIRED COMPONENTS ${PACKAGE_DEPENDENCIES})
catkin_package(
        LIBRARIES ${PROJECT_NAME}
        INCLUDE_DIRS include
        CATKIN_DEPENDS ${PACKAGE_DEPENDENCIES}
        DEPENDS yaml-cpp pinocchio)


set(SYSTEM_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${yaml_cpp_INCLUDE_DIRS})
include_directories(include ${SYSTEM_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE SRCS src/*.cpp)
add_library(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES} yaml-cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
