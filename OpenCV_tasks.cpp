
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

	Mat startImage = imread("space2.jpg");
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
	binarize(startImage, binary_image);
	windowName = "binary_4_lab";
	imwrite(windowName + "_MyImage.jpg", binary_image);
	namedWindow(windowName);
	imshow(windowName, binary_image);


	vector<uchar> row1{240, 240, 240};
	vector<uchar> row2{240, 240, 240};
	vector<uchar> row3{240, 240, 240};
	vector<vector<uchar>> mask{row1, row2, row3};

 	Mat dilatated;
	dilatation(binary_image, dilatated, mask, make_pair(1,1));

	windowName = "dilatated_4_lab";
	imwrite(windowName + "_MyImage.jpg", dilatated);
	namedWindow(windowName);
	imshow(windowName, dilatated);


 	Mat erozed;
	erosion(binary_image, erozed, mask, make_pair(1,1));

	windowName = "erozed_4_lab";
	imwrite(windowName + "_MyImage.jpg", erozed);
	namedWindow(windowName);
	imshow(windowName, erozed);

 	Mat open;
 	open_operation(binary_image, open, mask, make_pair(1,1));

	windowName = "open_4_lab";
	imwrite(windowName + "_MyImage.jpg", open);
	namedWindow(windowName);
	imshow(windowName, open);

 	Mat close;
 	close_operation(binary_image, close, mask, make_pair(1,1));

	windowName = "close_4_lab";
	imwrite(windowName + "_MyImage.jpg", close);
	namedWindow(windowName);
	imshow(windowName, close);


	waitKey(0);
	destroyAllWindows();

	return 0;
}

