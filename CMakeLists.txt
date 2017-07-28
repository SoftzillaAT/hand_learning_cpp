cmake_minimum_required (VERSION 2.6)
project(hand_learning)

find_package(V4R REQUIRED)
find_package( OpenCV 2.4 REQUIRED )
find_package( PCL REQUIRED )
find_package(libfreenect REQUIRED)

include_directories(include
	${V4R_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	"/usr/include/libusb-1.0/"
)

#define Debug level
# 0=No debug information
# 1=Some important debug information
# 2=All debug information
add_definitions(-DDEBUG_LEVEL=1)

add_executable(hand_learning main.cpp)

target_link_libraries(hand_learning
	${V4R_LIBS}
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${FREENECT_LIBRARIES}

)


install(TARGETS hand_learning RUNTIME DESTINATION bin)
