set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg_vclib/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(amcl_cuda LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.10)



find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(stb_image STATIC src/stb_image.cpp)
target_include_directories(stb_image PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(robotsim STATIC src/robot.cpp)
target_link_libraries(robotsim ${OpenCV_LIBS})
target_include_directories(robotsim PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(gpu_lidar STATIC src/gpu_lidar_2d.cpp src/gpu_lidar_kernels.cu)
target_include_directories(gpu_lidar PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET gpu_lidar PROPERTY CUDA_ARCHITECTURES 80 86)

add_executable(amcl_cuda src/main.cpp)
target_link_libraries(amcl_cuda stb_image gpu_lidar robotsim ${OpenCV_LIBS})
target_include_directories(amcl_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)

