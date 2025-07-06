add_subdirectory(modules)
add_subdirectory(classes)

add_library(utils STATIC utils.cpp)

target_link_libraries(utils PUBLIC opencv_world)