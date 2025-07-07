if(WIN32)
    add_library(dxdiag STATIC dxdiag.cpp)

    target_include_directories(
        dxdiag PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/helper
    )

    target_link_libraries(dxdiag PUBLIC opencv_world utils windowscodecs d3d11 dxguid)
endif()
