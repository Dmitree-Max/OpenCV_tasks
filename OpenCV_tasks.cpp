
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/core/saturate.hpp>

#include "OpenCV_tasks.h"
#include "borderFounder.h"

using namespace cv;
using namespace std;


static void showImage(const Mat& image, string windowName) {
	imwrite(windowName + "_MyImage.jpg", image);
	namedWindow(windowName);
	imshow(windowName, image);
}


static void showChangedImage(const Mat& image, function<void(const Mat&, Mat&)> f, string windowName) {
	Mat newMat = image.clone();
	f(image, newMat);

	imwrite(windowName + "_MyImage.jpg", newMat);
	namedWindow(windowName);
	imshow(windowName, newMat);
}


int main(int argc, char** argv) {

	Mat startImage = imread("forborder.jpg");
	if (startImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	Mat grad_x;
	Mat grad_y;
	Mat abs_grad_x, abs_grad_y;

    cvtColor(startImage, startImage, COLOR_BGR2GRAY);

    Mat distImage = startImage.clone();


    Sobel(startImage, grad_y, CV_16U, 0, 1);
    Sobel(startImage, grad_x, CV_16U, 1, 0);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, distImage);

	showImage(startImage, "Original image");
	showImage(distImage, "Original Sobel");

	showChangedImage(startImage, my_sobel, "Custom Sobel");
	showChangedImage(startImage, my_roberts, "Custom Roberts");
	showChangedImage(startImage, my_previt, "Custom Previt");


	waitKey(0);
	destroyAllWindows();

	return 0;
}

