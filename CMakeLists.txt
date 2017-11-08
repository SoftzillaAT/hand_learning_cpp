cmake_minimum_required (VERSION 2.6)
project(hand_learning)

find_package(V4R REQUIRED)
find_package( OpenCV 2.4 REQUIRED )
find_package( PCL REQUIRED )
find_package(libfreenect REQUIRED)

include_directories(include
	./include
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
add_library(FaceDetection FaceDetection.cpp)
add_library(ObjSegmentation ObjSegmentation.cpp)
add_library(PclManipulation PclManipulation.cpp)
add_library(ColorHistogram ColorHistogram.cpp)
add_library(adaptiveskindetector adaptiveskindetector.cpp)
add_library(Camera Camera.cpp)

add_executable(hand_learning main.cpp)

target_link_libraries(hand_learning
	${V4R_LIBS}
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${FREENECT_LIBRARIES}
	FaceDetection
	ObjSegmentation
	PclManipulation
	Camera
  ColorHistogram
  adaptiveskindetector
)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

configure_file(./data/haarcascade_frontalface_alt.xml ./data/haarcascade_frontalface_alt.xml COPYONLY)
configure_file(./data/haarcascade_frontalface_default.xml ./data/haarcascade_frontalface_default.xml COPYONLY)
configure_file(./data/haarcascade_eye_tree_eyeglasses.xml ./data/haarcascade_eye_tree_eyeglasses.xml COPYONLY)
configure_file(./data/haarcascade_eye.xml ./data/haarcascade_eye.xml COPYONLY)
configure_file(./data/lbpcascade_frontalface.xml ./data/lbpcascade_frontalface.xml COPYONLY)



install(TARGETS hand_learning RUNTIME DESTINATION bin)
