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

add_executable(hand_learning main.cpp)

target_link_libraries(hand_learning
	${V4R_LIBS}
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${FREENECT_LIBRARIES}

)


install(TARGETS hand_learning RUNTIME DESTINATION bin)
