cmake_minimum_required(VERSION 3.30)

include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC GREATER 1)
    set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} ${NPROC})
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/runtime")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT environment variable must be set before running CMake.")
endif()

if(DEFINED ENV{VCPKG_ROOT})
  message(STATUS "VCPKG_ROOT: $ENV{VCPKG_ROOT}")
  file(TO_CMAKE_PATH $ENV{VCPKG_ROOT} VCPKG_ROOT_PATH)
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows")
    message(STATUS "VCPKG TARGET TRIPLET: ${VCPKG_TARGET_TRIPLET}")
  endif()
  set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT_PATH}/scripts/buildsystems/vcpkg.cmake")
  set(VCPKG_MANIFEST_INSTALL OFF)
else()
  message(STATUS "VCPKG_ROOT not defined")
endif()

project(agent_live LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

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
  target_link_libraries(${PROJECT_NAME} PRIVATE App)
endif()

add_executable(agent_webcam agent_webcam.cpp)
target_link_libraries(agent_webcam PRIVATE yolo dxdiag)

add_executable(agent_screenshot agent_screenshot.cpp)
target_link_libraries(agent_screenshot PRIVATE screenshot)

function(setup_runtime_dll_dir target_name)
    set(RUNTIME_DLL_DIR "${CMAKE_CURRENT_BINARY_DIR}/runtime")

    add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DLL_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target_name}> "${RUNTIME_DLL_DIR}"
            COMMAND_EXPAND_LISTS
    )
endfunction()

foreach(app_target agent_webcam agent_screenshot ${PROJECT_NAME})
    setup_runtime_dll_dir(${app_target})
endforeach()

#NOTE TO SELF:
#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")