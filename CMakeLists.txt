cmake_minimum_required(VERSION 3.16)
project(esca-vision VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(SDBUSCPP REQUIRED IMPORTED_TARGET sdbus-c++)

# TensorFlow Lite installs libtensorflow-lite.so and its headers under
# tensorflow/lite/, with no CMake config package, so look it up by name.
find_library(TFLITE_LIB NAMES tensorflow-lite REQUIRED)
find_path(TFLITE_INCLUDE_DIR tensorflow/lite/interpreter.h REQUIRED)

# libtensorflow-lite.so calls into Abseil's logging but does not record the
# dependency, so the final link has to supply it: LogMessage lives in
# log_internal_message and kCharNull in log_internal_nullguard.
find_library(ABSL_LOG_MESSAGE_LIB NAMES absl_log_internal_message REQUIRED)
find_library(ABSL_LOG_NULLGUARD_LIB NAMES absl_log_internal_nullguard REQUIRED)

# From the camera service: the client library and its shared-memory reader.
find_library(ESCA_CAMCLIENT_LIB NAMES esca-camclient REQUIRED)
find_path(ESCA_CAM_INCLUDE_DIR esca/camera/client.hpp REQUIRED)

# --- libesca-visiondetect: the model, shared by the daemon and the tool -----
# Static, because it is an implementation detail of this project: nothing
# outside it should link TensorFlow Lite to get at our detections.
add_library(esca-visiondetect STATIC src/common/detector.cpp)
target_include_directories(esca-visiondetect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${TFLITE_INCLUDE_DIR})
target_link_libraries(esca-visiondetect PUBLIC
    ${TFLITE_LIB} ${ABSL_LOG_MESSAGE_LIB} ${ABSL_LOG_NULLGUARD_LIB} Threads::Threads)

# --- libesca-visionclient: what applications link ---------------------------
add_library(esca-visionclient SHARED src/client/client.cpp)
target_include_directories(esca-visionclient PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(esca-visionclient PUBLIC PkgConfig::SDBUSCPP Threads::Threads)
set_target_properties(esca-visionclient PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

# --- esca-visiond: the service ----------------------------------------------
add_executable(esca-visiond src/daemon/main.cpp src/daemon/service.cpp)
target_include_directories(esca-visiond PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common ${ESCA_CAM_INCLUDE_DIR})
target_link_libraries(esca-visiond PRIVATE
    esca-visiondetect PkgConfig::SDBUSCPP ${ESCA_CAMCLIENT_LIB} Threads::Threads)

# --- esca-vision-detect: the benchmark, no service involved -----------------
add_executable(esca-vision-detect src/tools/detect.cpp)
target_include_directories(esca-vision-detect PRIVATE ${ESCA_CAM_INCLUDE_DIR})
target_link_libraries(esca-vision-detect PRIVATE esca-visiondetect ${ESCA_CAMCLIENT_LIB})

# --- esca-visionctl: the service, no display involved -----------------------
add_executable(esca-visionctl src/tools/visionctl.cpp)
target_link_libraries(esca-visionctl PRIVATE esca-visionclient)

install(TARGETS esca-visionclient LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS esca-visiond RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR})
install(TARGETS esca-vision-detect esca-visionctl RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/esca DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES data/vn.sparc.esca.Vision1.conf
        DESTINATION ${CMAKE_INSTALL_DATADIR}/dbus-1/system.d)
install(FILES data/esca-visiond.service
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/systemd/system)
