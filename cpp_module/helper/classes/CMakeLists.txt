add_library(screenshot STATIC Screenshot.cpp)
add_library(yolo STATIC Yolo.cpp)

target_include_directories(
    yolo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/helper
)

target_include_directories(
    screenshot PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/helper
)

target_link_libraries(yolo PUBLIC utils)
target_link_libraries(screenshot PUBLIC yolo utils)

if(WIN32)
    target_link_libraries(screenshot PUBLIC dxdiag)
endif()