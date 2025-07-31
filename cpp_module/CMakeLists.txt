cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT environment variable must be set before running CMake.")
endif()

if(DEFINED ENV{VCPKG_ROOT})
  message(STATUS "VCPKG_ROOT: $ENV{VCPKG_ROOT}")
  set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
  #set(VCPKG_MANIFEST_INSTALL OFF)
else()
  message(STATUS "VCPKG_ROOT not defined")
endif()

#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
project(agent_live LANGUAGES CXX)
enable_language(CUDA OPTIONAL)

find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
else()
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV Include Dir: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV Libs: ${OpenCV_LIBS}")
endif()

add_subdirectory(helper)

if(WIN32)
  add_executable(${PROJECT_NAME} agent_live.cpp)
  target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper)
  target_link_libraries(${PROJECT_NAME} PRIVATE dxdiag yolo d3d11 dxguid utils)
endif()

if(UNIX)
  add_executable(agent_webcam agent_webcam.cpp)
  target_link_libraries(agent_webcam PRIVATE yolo)

  # agent_screenshot executable
  add_executable(agent_screenshot agent_screenshot.cpp)
  target_link_libraries(agent_screenshot PRIVATE screenshot)
endif()

add_executable(cv_test test.cpp)
target_link_libraries(cv_test opencv_core opencv_imgproc opencv_highgui)

# add_executable(yoloPro yolo_Test.cpp)
# target_link_libraries(yoloPro opencv_core opencv_dnn opencv_cudaarithm opencv_videoio opencv_highgui opencv_imgproc)