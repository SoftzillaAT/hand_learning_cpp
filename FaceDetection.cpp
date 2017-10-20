#include "FaceDetection.h"


FaceDetection::FaceDetection(Mat image)
{
				_image = image;
				face_cascade_name = "./data/haarcascade_frontalface_alt.xml";
				eyes_cascade_name = "./data/haarcascade_eye.xml";

}

void FaceDetection::showResult()
{
				//-- Show what you got
				imshow( "Face detection", _image );
}

bool FaceDetection::detectFace()
{
	_face.release();
				//-- 1. Load the cascades
				if( !face_cascade.load( face_cascade_name ) )
				{ 
								cout << "Error Loading xml" << face_cascade_name << endl; 
								return false;
				}

				if( !eyes_cascade.load( eyes_cascade_name ) )
				{ 
								cout << "Error Loading xml" << eyes_cascade_name << endl; 
								return false; 
				}

				std::vector<Rect> faces;
				Mat frame_gray;

				cvtColor( _image, frame_gray, CV_BGR2GRAY );
				equalizeHist( frame_gray, frame_gray );

				//-- Detect faces
				face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 
												0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

				for( size_t i = 0; i < faces.size(); i++ )
				{
								faces[i].x = faces[i].x + faces[i].width * 0.1;
								faces[i].width = faces[i].width*0.8;
								faces[i].y = faces[i].y + faces[i].height * 0.05;
								faces[i].height = faces[i].height*0.9;
								_face = _image( faces[i] );
								
								cv::imshow("FaceROI", _face);

								Point center( faces[i].x + faces[i].width*0.5, 
																faces[i].y + faces[i].height*0.5 );

								ellipse( _image, center, Size( faces[i].width*0.5, 
																				faces[i].height*0.5), 0, 0, 360, 
																Scalar( 255, 0, 255 ), 4, 8, 0 );

								// skip other faces and eye detection
								break;
								Mat faceROI = frame_gray ( faces[i] );
								std::vector<Rect> eyes;
								//-- In each face, detect eyes
								eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 
																0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

								for( size_t j = 0; j < eyes.size(); j++ )
								{
												Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, 
																				faces[i].y + eyes[j].y + eyes[j].height*0.5 );
												int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
												circle( _image, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
								}
				}
				
				return !_face.empty();
}

void FaceDetection::calcHistogram()
{
				Mat src, hsv;
				src = _image;
				cvtColor(src, hsv, CV_BGR2HSV);

				// Quantize the hue to 30 levels
				// and the saturation to 32 levels
				int hbins = 30, sbins = 32;
				int histSize[] = {hbins, sbins};
				// hue varies from 0 to 179, see cvtColor
				float hranges[] = { 0, 180 };
				// saturation varies from 0 (black-gray-white) to
				// 255 (pure spectrum color)
				float sranges[] = { 0, 256 };
				const float* ranges[] = { hranges, sranges };
				MatND hist;
				// we compute the histogram from the 0-th and 1-st channels
				int channels[] = {0, 1};

				calcHist( &hsv, 1, channels, Mat(), // do not use mask
												hist, 2, histSize, ranges,
												true, // the histogram is uniform
												false );
				double maxVal=0;
				minMaxLoc(hist, 0, &maxVal, 0, 0);

				int scale = 10;
				Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);

				for( int h = 0; h < hbins; h++ )
								for( int s = 0; s < sbins; s++ )
								{
												float binVal = hist.at<float>(h, s);
												int intensity = cvRound(binVal*255/maxVal);
												rectangle( histImg, Point(h*scale, s*scale),
																				Point( (h+1)*scale - 1, (s+1)*scale - 1),
																				Scalar::all(intensity),
																				CV_FILLED );
								}

				namedWindow( "Source", 1 );
				imshow( "Source", src );

				namedWindow( "H-S Histogram", 1 );
				imshow( "H-S Histogram", histImg );
				waitKey();
}

std::vector< cv::Vec3b > FaceDetection::getSkinPoints()
{
	_skin_points.clear();
				if (!_face.empty())
				{
								// convert to hsv
								Mat hsv_face;
								cv::cvtColor(_face, hsv_face, CV_BGR2HSV);
								for (int x = 0; x < hsv_face.rows; x++)
								{
												for (int y = 0; y < hsv_face.cols; y++)
												{
																_skin_points.push_back( hsv_face.at<cv::Vec3b>(x,y) );
												}
								}
				}

				return _skin_points;
}

std::vector<cv::Point3f> FaceDetection::getHsvCylinder()
{
	std::vector<cv::Point3f> result;

	std::vector<cv::Vec3b> points = getSkinPoints();

	for (int i = 0; i < points.size(); i++)
	{
		uint8_t h = points[i][0];
		uint8_t s = points[i][1];
		uint8_t v = points[i][2];

		float alpha = (float)h/90.0 * M_PI;
		float x = cos(alpha) * s;
		float y = sin(alpha) * s;
		float z = v;
		Point3f p(x,y,z);
		result.push_back(p);
	}

	return result;
}


