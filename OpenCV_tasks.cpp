
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/core/saturate.hpp>

#include "OpenCV_tasks.h"
#include "histogramManipulation.h"

using namespace cv;
using namespace std;


static void showImage(const Mat& image, string windowName) {
	imwrite(windowName + "_MyImage.jpg", image);
	namedWindow(windowName);
	showHistogram(image, windowName + "_hist");
	imshow(windowName, image);
}

static void showChangedImage(const Mat& image, function<void(const Mat&, Mat&, float)> f, float coeffecient, string windowName) {
	Mat newMat = image.clone();
	f(image, newMat, coeffecient);

	imwrite(windowName + "_MyImage.jpg", newMat);
	showHistogram(newMat, windowName + "_hist");
	namedWindow(windowName);
	imshow(windowName, newMat);
}

static void showChangedImage(const Mat& image, function<void(const Mat&, Mat&)> f, string windowName) {
	Mat newMat = image.clone();
	f(image, newMat);

	imwrite(windowName + "_MyImage.jpg", newMat);
	showHistogram(newMat, windowName + "_hist");
	namedWindow(windowName);
	imshow(windowName, newMat);
}

static void showChangedImage(const Mat& image, const Mat& secondImage, function<void(const Mat&, const Mat&, Mat&)> f, string windowName) {
	Mat newMat = image.clone();
	f(image, secondImage, newMat);

	imwrite(windowName + "_MyImage.jpg", newMat);
	showHistogram(newMat, windowName + "_hist");
	showHistogram(secondImage, "dist_hist");
	namedWindow(windowName);
	imshow(windowName, newMat);
}

int main(int argc, char** argv) {

	Mat startImage = imread("sandwitch.jpg");
	if (startImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	Mat grad_x;
	Mat grad_y;
	Mat abs_grad_x, abs_grad_y;
//	Mat distImage;


    cvtColor(startImage, startImage, COLOR_BGR2GRAY);

    String distFileName = "dark.jpg";
	Mat distImage = imread(distFileName);
	if (distImage.empty()) {
		cout << "Could not open or find the image " << distFileName << endl;
		return -1;
	}
	cvtColor(distImage, distImage, COLOR_BGR2GRAY);

    showImage(startImage, "The sandwitch");
    showChangedImage(startImage, equalizeImage, "Equalized sandwitch");
    showChangedImage(startImage, linearyChangeImage, 0.2, "Lin sandwitch");
    showChangedImage(startImage, distImage, histogramApplication, "App sandwitch");


//    Sobel(startImage, grad_y, CV_16U, 0, 1);
//    Sobel(startImage, grad_x, CV_16U, 1, 0);
//
//    convertScaleAbs(grad_x, abs_grad_x);
//    convertScaleAbs(grad_y, abs_grad_y);
//
//    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, distImage);
//
//	showImage(startImage, "The cars");
//
//	showImage(distImage, "The cars borders");


	waitKey(0);
	destroyAllWindows();

	return 0;
}

