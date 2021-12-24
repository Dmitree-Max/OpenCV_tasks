
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/core/saturate.hpp>

#include "morphology.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv) {

	Mat startImage = imread("slon3.jpg");
	if (startImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

    cvtColor(startImage, startImage, COLOR_BGR2GRAY);


    string windowName = "start_image_4_lab";
	imwrite(windowName + "_MyImage.jpg", startImage);
	namedWindow(windowName);
	imshow(windowName, startImage);

	Mat binary_image;
	binarize(startImage, binary_image, 128);
	windowName = "binary_128";
	imwrite(windowName + "_MyImage.jpg", binary_image);
	namedWindow(windowName);
	imshow(windowName, binary_image);

	int mean = find_mean_value(startImage);
	std::cout << "mean: " << mean << std::endl;
	binarize(startImage, binary_image, mean);
	windowName = "binary_mean";
	imwrite(windowName + "_MyImage.jpg", binary_image);
	namedWindow(windowName);
	imshow(windowName, binary_image);


	int otsu = find_treshold_otsu(startImage);
	std::cout << "otsu: " << otsu << std::endl;
	binarize(startImage, binary_image, otsu);
	windowName = "binary_otsu";
	imwrite(windowName + "_MyImage.jpg", binary_image);
	namedWindow(windowName);
	imshow(windowName, binary_image);

	waitKey(0);
	destroyAllWindows();

	return 0;
}

