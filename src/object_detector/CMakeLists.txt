cmake_minimum_required(VERSION 3.12)

project(yolov5-object-detector-lib
  VERSION 1.0.0
  DESCRIPTION "YOLOv5 ObjectDetector library"
  LANGUAGES CXX
)
# Link OpenMP
find_package(OpenMP REQUIRED)

link_directories(/usr/lib/x86_64-linux-gnu)
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
  message(STATUS "OpenCV library status:")
  message(STATUS "  version: ${OpenCV_VERSION}")
  message(STATUS "  include path: ${OpenCV_INCLUDE_DIRS}")
else ()
  message(FATAL_ERROR "Could not find OpenCV")
endif ()
# Find GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0)
message(STATUS "GStreamer library status:")
message(STATUS "  version: ${GSTREAMER_VERSION}")
message(STATUS "  include path: ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "  library path: ${GSTREAMER_LIBRARY_DIRS}")

set(Torch_DIR ${CMAKE_SOURCE_DIR}/libtorch_v1-6-0/share/cmake/Torch/)
find_package(Torch PATHS ${Torch_DIR} REQUIRED)
if (TORCH_FOUND)
  message(STATUS "Torch library status:")
  message(STATUS "  version: ${Torch_VERSION}")
  message(STATUS "  include path: ${TORCH_INCLUDE_DIRS}")
else ()
  message(FATAL_ERROR "Could not find Torch")
endif ()

add_library(object_detector STATIC
  ${PROJECT_SOURCE_DIR}/src/object_detector.cpp
     ${PROJECT_SOURCE_DIR}/src/RTSPcam.cpp
)

target_compile_features(object_detector PRIVATE cxx_std_17)

target_include_directories(object_detector PUBLIC
  ${PROJECT_SOURCE_DIR}/include/
  ${OpenCV_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)


target_link_libraries(object_detector PUBLIC
  ${OpenCV_LIBS}
  ${GSTREAMER_LIBRARIES}
  ${GSTREAMER_APP_LIBRARIES}
  ${GSTREAMER_VIDEO_LIBRARIES}
  ${TORCH_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(object_detector PUBLIC OpenMP::OpenMP_CXX)
endif()
