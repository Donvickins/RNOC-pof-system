add_subdirectory(modules)
add_subdirectory(classes)

add_library(utils STATIC utils.cpp)
add_library(App STATIC App.cpp)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(utils PUBLIC opencv_core opencv_cudaarithm opencv_imgcodecs opencv_dnn opencv_videoio opencv_highgui)

target_link_libraries(App PUBLIC dxdiag yolo utils)