cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT environment variable must be set before running CMake.")
endif()

if(DEFINED ENV{VCPKG_ROOT})
  message(STATUS "VCPKG_ROOT: $ENV{VCPKG_ROOT}")
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static")
    message(STATUS "VCPKG TARGET TRIPLET: ${VCPKG_TARGET_TRIPLET}")
  endif()
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
  set(VCPKG_MANIFEST_INSTALL OFF)
else()
  message(STATUS "VCPKG_ROOT not defined")
endif()

project(agent_live LANGUAGES CXX)

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

add_executable(agent_webcam agent_webcam.cpp)
target_link_libraries(agent_webcam PRIVATE yolo)

add_executable(agent_screenshot agent_screenshot.cpp)
target_link_libraries(agent_screenshot PRIVATE screenshot)


#NOTE TO SELF:
#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")