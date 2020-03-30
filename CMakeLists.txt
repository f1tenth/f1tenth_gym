cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(racecar_simulator_standalone)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g") 
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


find_package(Eigen3 REQUIRED)
include(FindProtobuf)
find_package(Protobuf REQUIRED)
find_package(PkgConfig)
# find_package(OpenCV)
pkg_check_modules(PC_ZeroMQ QUIET zmq)
find_path(ZeroMQ_INCLUDE_DIR
          NAMES zmq.hpp
          PATHS ${PC_ZeroMQ_INCLUDE_DIRS} 
                "/usr/local/include"
          )
find_library(ZeroMQ_LIBRARY
             NAMES zmq
             PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
             )

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS sim_requests.proto)
PROTOBUF_GENERATE_PYTHON(PROTO_PY sim_requests.proto)
message("ZMQ_INCLUDE_DIR = ${ZeroMQ_INCLUDE_DIR}")
message ("PROTO_SRCS = ${PROTO_SRCS}")
message ("PROTO_HDRS = ${PROTO_HDRS}")
message ("PROTO_PY = ${PROTO_PY}")

# dummy target so it triggers python generation
add_custom_target(dummy ALL 
                  DEPENDS ${PROTO_PY})

# include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(
    ${PROTOBUF_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

file(GLOB SOURCE_FILES
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cc"
        )

# executable for simulator server
add_executable(sim_server ${SOURCE_FILES}
                          ${PROTO_SRCS}
                          ${PROTO_HDRS})

target_include_directories(sim_server PUBLIC 
                           ${ZeroMQ_INCLUDE_DIR}
                           include
                           ${PROTOBUF_INCLUDE_DIR}
                           ${CMAKE_CURRENT_BINARY_DIR}
                           )
target_link_libraries(sim_server PUBLIC
                      ${ZeroMQ_LIBRARY}
                      ${PROTOBUF_LIBRARY})
                      # ${OpenCV_LIBS})

# target_link_libraries(task_worker PUBLIC
#                     ${ZeroMQ_LIBRARY}
#                     ${PROTOBUF_LIBRARY})
#                     # ${OpenCV_LIBS})
