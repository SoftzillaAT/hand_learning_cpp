#include "ObjSegmentation.h"


ObjSegmentation::ObjSegmentation(Mat image)
{
				_image = image;
}

void ObjSegmentation::grabCut()
{
	cv::Rect rectangle(151 ,229 ,170, 130);

	cv::Mat result; // segmentation result (4 possible values)
	cv::Mat bgModel,fgModel; // the models (internally used)

	// GrabCut segmentation
	cv::grabCut(_image,    // input image
	            result,   // segmentation result
							rectangle,// rectangle containing foreground
							bgModel,fgModel, // models
							2,        // number of iterations
							cv::GC_INIT_WITH_RECT); // use rectangle

	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	cv::Mat foreground(_image.size(),CV_8UC3,cv::Scalar(255,255,255));
	_image.copyTo(foreground, result); // bg pixels not copied

	imshow("Segmentation", foreground);
}


void ObjSegmentation::setSkinMask(std::vector<cv::Vec3b> ref_pixels, float max_dist1, float max_dist2)
{

	cv::Mat image_hsv;
	cv::cvtColor(_image, image_hsv, CV_BGR2HSV);


	skin_mask = Mat::zeros(_image.rows, _image.cols, CV_8UC1);
	

	cout << "Pixel: " << image_hsv.at<cv::Vec3b>(111,184) << endl;
	for (int i = 0; i < ref_pixels.size(); i++)
	{
		cv::Vec3b ref = ref_pixels[i];
		uchar range = 20;
		uchar range1 = 40;
		uchar range2 = 80;
		uchar l0 = (ref.val[0] - range < 0) ? 0 : ref.val[0] - range;
		uchar l1 = (ref.val[1] - range1 < 0) ? 0 : ref.val[1] - range1;
		uchar l2 = (ref.val[2] - range2 < 0) ? 0 : ref.val[2] - range2;
		
		uchar h0 = (ref.val[0] + range > 179) ? 179 : ref.val[0] + range;
		uchar h1 = (ref.val[1] + range1 > 255) ? 255 : ref.val[1] + range1;
		uchar h2 = (ref.val[2] + range2 > 255) ? 255 : ref.val[2] + range2;

		
		cv::Vec3b low(l0, l1, l2);
		cv::Vec3b high(h0, h1 , h2);

		//cv::Scalar low(0,10,50);
		//cv::Scalar high(170, 50, 100);
		
		cout << "Low: " << low << " | High: " << high << endl;
		Mat mask;
		
		cv::inRange(image_hsv, low, high, mask);
		cv::bitwise_or(skin_mask, mask, skin_mask);
	}
/*
	for (int x = 0; x < image_hsv.cols; x++)
	{
		for (int y = 0; y < image_hsv.rows; y++)
		{
			cv::Point2d p(x,y);
			cv::Vec3b color = image_hsv.at<cv::Vec3b>(p);
			for (int i = 0; i < ref_pixels.size(); i++)
			{
				float dist = cv::norm(color, ref_pixels[i], cv::NORM_L2);
				//cout << dist << " | ";
				if (dist <= max_dist1)
				{
					skin_mask.at<uchar>(p) = 255;
					break;
				}
			}
		}
	}
	cout << endl;
	*/
	cv::imshow("Hand Mask before hysterese", skin_mask);
	applyHystereses(image_hsv, max_dist2);
	cv::imshow("Hand Mask after hysterese", skin_mask);
	Mat dilate_mask;
	Mat kernel = Mat::ones(9,9,CV_8UC1);
	cv::morphologyEx( skin_mask, dilate_mask, 3, kernel );
	cv::imshow("dilating", dilate_mask);
	//cv::dilate(skin_mask, dilate_mask,
	Mat skin_result;
	_image.copyTo(skin_result, ~dilate_mask);
	cv::imshow("Skin result without pcl", skin_result);
	skin_mask = dilate_mask;

}


void ObjSegmentation::applyHystereses(cv::Mat image_hsv, float max_dist)
{

	std::vector<cv::Point2d> queue; // queue of skin pixels
	std::vector<cv::Point2d> visited; // queue of visited pixels

	for (int x = 0; x < image_hsv.cols; x++)
	{
		for (int y = 0; y < image_hsv.rows; y++)
		{
			cv::Point2d p(x,y);
			if (skin_mask.at<uchar>(p)>0)
			{
				queue.push_back(p);
			}
		}
	}

	while (queue.size() > 0)
	{
		cv::Point2d p = queue[0];
		queue.erase(queue.begin());
		
		if (std::find(visited.begin(), visited.end(), p) != visited.end())
			continue;

		visited.push_back(p);
		
		// iterate through neighbours
		for (int i = 0; i < 8; i++)
		{
			int x = p.x;
			int y = p.y;

			if (i == 0) { x = x - 1; y = y - 1; }
			if (i == 1) { y = y - 1; }
			if (i == 2) { x = x + 1; y = y - 1; }
			
			if (i == 3) { x = x - 1; }
			if (i == 4) { x = x + 1; }

			if (i == 5) { x = x - 1; y = y + 1;}
			if (i == 6) { y = y + 1; }
			if (i == 7) { x = x + 1; y = y + 1; }

			if ( x <= 0 || x >= skin_mask.cols || y <= 0 || y >= skin_mask.rows )
				continue;

			cv::Point2d p2(x, y);
			cv::Vec3b color = image_hsv.at<cv::Vec3b>(p2);
			cv::Vec3b ref_color = image_hsv.at<cv::Vec3b>(p);

			float dist = norm(color, ref_color, cv::NORM_L2);
			
			if (dist <= max_dist)
			{
					skin_mask.at<uchar>(p2) = 255;
					queue.push_back(p2);
			}
		}	
	}

}

void ObjSegmentation::applyMask(PointCloud<PointXYZRGB>::Ptr& cloud)
{
	const float bad_point = std::numeric_limits<float>::quiet_NaN();

	for (unsigned v = 0; v < cloud->height; v++)
	{
					for (unsigned u = 0; u < cloud->width; u++)
					{

									pcl::PointXYZRGB &pt = (*cloud)(u,v);
									if( skin_mask.at<bool>(v, u) > 0 )
									{
													pt.x = bad_point;
													pt.y = bad_point;
													pt.z = bad_point;
													pt.r = 0;
													pt.g = 0;
													pt.b = 0;
									}
					}
	}
}
