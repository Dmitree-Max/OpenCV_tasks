
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/core/saturate.hpp>

#include "OpenCV_tasks.h"
#include "noise.h"

using namespace cv;
using namespace std;


static void showImage(const Mat& image, string windowName) {
	imwrite(windowName + "_MyImage.jpg", image);
	namedWindow(windowName);
	imshow(windowName, image);
}

static void showChangedImage(const Mat& image, const Mat& noised_image,
		function<void(const Mat&,const Mat&,Mat&)> f, string windowName) {
	Mat cleaned_image = image.clone();
	f(image, noised_image, cleaned_image);
	std::cout << "Difference between start image and " << windowName << " is "
			<< mean_absolute_difference(image, cleaned_image) << std::endl;
	imwrite(windowName + "_MyImage.jpg", cleaned_image);
	namedWindow(windowName);
	imshow(windowName, cleaned_image);
}


int main(int argc, char** argv) {
	Mat startImage = imread("pigs.jpg");
	if (startImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

    cvtColor(startImage, startImage, COLOR_BGR2GRAY);
    showImage(startImage, "The start image");

	Mat noisedImage = startImage.clone();
	apply_noise(startImage, noisedImage);
	showImage(noisedImage, "Noised");
	std::cout << "Difference between start image and noised is "
			<< mean_absolute_difference(startImage, noisedImage) << std::endl;

    showChangedImage(startImage, noisedImage, gaussian_method, "Gauss");


	waitKey(0);
	destroyAllWindows();

	return 0;
}

