set(vcpkg "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
if(NOT CMAKE_TOOLCHAIN_FILE AND EXISTS "${vcpkg}")
  set(CMAKE_TOOLCHAIN_FILE "${vcpkg}"
      CACHE FILEPATH "CMake toolchain file")
  message(STATUS "vcpkg toolchain found: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(WARN "vcpkg toolchain not found")
endif()

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

add_library(particle_filter src/pf.cpp src/pf_kernels.cu)
target_include_directories(particle_filter PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET particle_filter PROPERTY CUDA_ARCHITECTURES 80 86)

add_library(gpu_lidar STATIC src/gpu_lidar_2d.cpp src/gpu_lidar_kernels.cu)
target_include_directories(gpu_lidar PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET gpu_lidar PROPERTY CUDA_ARCHITECTURES 80 86)

add_executable(amcl_cuda src/main.cpp)
target_link_libraries(amcl_cuda stb_image gpu_lidar robotsim particle_filter ${OpenCV_LIBS})
target_include_directories(amcl_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)

