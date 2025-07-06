cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
endif()

add_subdirectory(helper)

if(WIN32)
  add_executable(${PROJECT_NAME} agent_live.cpp)
  target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper)
  target_link_libraries(${PROJECT_NAME} PRIVATE dxdiag yolo d3d11 dxguid utils)
endif()

add_executable(agent_webcam agent_webcam.cpp)
target_include_directories(agent_webcam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper)
target_link_libraries(agent_webcam PRIVATE yolo utils)

add_executable(agent_screenshot agent_screenshot.cpp)
target_link_libraries(agent_screenshot PRIVATE screenshot yolo utils)