
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/core/saturate.hpp>

#include "OpenCV_tasks.h"

using namespace cv;
using namespace std;


static void showImage(const Mat& image, string windowName) {
	namedWindow(windowName);
	imshow(windowName, image);
}

static void showChangedImage(const Mat& image, function<void(const Mat&, Mat&, float)> f, float coeffecient, string windowName) {
	Mat newMat = image.clone();
	f(image, newMat, coeffecient);

	namedWindow(windowName);
	imshow(windowName, newMat);
}

static void showChangedImage(const Mat& image, function<void(const Mat&, Mat&)> f, string windowName) {
	Mat newMat = image.clone();
	f(image, newMat);

	namedWindow(windowName);
	imshow(windowName, newMat);
}

static void showChangedImage(const Mat& image, const Mat& secondImage, function<void(const Mat&, const Mat&, Mat&)> f, string windowName) {
	Mat newMat = image.clone();
	f(image, secondImage, newMat);

	namedWindow(windowName);
	imshow(windowName, newMat);
}

int main(int argc, char** argv) {

	Mat startImage = imread("sandwitch.jpg");
	if (startImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

    cvtColor(startImage, startImage, COLOR_BGR2GRAY);

    String distFileName = "dark.jpg";
	Mat distImage = imread(distFileName);
	if (distImage.empty()) {
		cout << "Could not open or find the image " << distFileName << endl;
		return -1;
	}
	cvtColor(distImage, distImage, COLOR_BGR2GRAY);

    showImage(startImage, "The Space");
    showChangedImage(startImage, equalizeImage, "Equalized Space");
    showChangedImage(startImage, linearyChangeImage, 0.05, "Lin Space");
    showChangedImage(startImage, distImage, histogramApplication, "App Space");

	waitKey(0);
	destroyAllWindows();

	return 0;
}


void equalizeImage(const Mat& input, Mat& output) {
	auto srcHist = getCumulativeHistogramFromImage(input);
	unique_ptr<vector<float>> distHist (new vector<float> (srcHist->size()));

	for (size_t i = 0 ; i < distHist->size(); i++) {
		(*distHist)[i] = (float)i;
	}
	distHist = move(normalizeHist(*distHist));

	auto cMap = getColorMap(*distHist, *srcHist);
	applyColorMapToImage(output, *cMap);
}


template <typename T>
unique_ptr<vector<float> > normalizeHist(const vector<T>& histogram) {
	unique_ptr<vector<float>> result (new vector<float> (histogram.size()));

	int max_el = *max_element(histogram.begin(), histogram.end());
	for (size_t i = 0; i < histogram.size(); ++i) {
		(*result)[i] = histogram[i] / (float)max_el;
	}

	float s = accumulate(result->begin(), result->end(), 0.0);
	for_each(result->begin(), result->end(), [s](float& el){ el /= s; });

	return result;
}


void histogramApplication(const Mat& input, const Mat& dist, Mat& output) {
	auto distHist = getCumulativeHistogramFromImage(dist);
	auto srcHist = getCumulativeHistogramFromImage(input);

	auto cMap = getColorMap(*distHist, *srcHist);
	applyColorMapToImage(output, *cMap);
}

void linearyChangeImage(const Mat& input, Mat& output, float coeffecient) {
	double minVal;
	double maxVal;

	minMaxLoc( input, &minVal, &maxVal, nullptr, nullptr );

	output = input.clone();

	if (abs(minVal - maxVal) < pow(10, -6)) {
		return;
	}

	double middle = (maxVal + minVal) / 2;
	for(int y = 0; y < input.rows; y++)
	{
	    for(int x = 0; x < input.cols; x++)
	    {
	    	unsigned char oldColor = input.at<uchar>(y,x);
	    	output.at<uchar>(y,x) =
	    			saturate_cast<unsigned char>(oldColor * (1 + coeffecient)
	    					- middle * coeffecient);

	    }

	}
}



unique_ptr<map<int, int>> getColorMap(const vector<float>& histogramDist,
			const vector<float>& histogramSrc) {
	 	 assert(histogramDist.size() == histogramSrc.size());
	 	 assert(abs(accumulate(histogramDist.begin(), histogramDist.end(), 0.0) - 1) < pow(10, -6));
	 	 assert(abs(accumulate(histogramSrc.begin(), histogramSrc.end(), 0.0)  - 1) < pow(10, -6));

	 	 unique_ptr<map<int, int>> result(new map<int,int>());

	 	 size_t current_dist = 0;

	 	 for (size_t i = 0; i < histogramSrc.size();) {
	 		 if (current_dist == histogramSrc.size() - 1) {
	 			 result->emplace(make_pair(i, current_dist));
	 			 ++i;
	 			 continue;
	 		 }
	 		 if (histogramSrc[i]  > histogramDist[current_dist + 1]) {
	 			 ++current_dist;
	 			 continue;
	 		 }

	 		 if (abs(histogramSrc[i] - histogramDist[current_dist])
	 		 	 	 < abs(histogramSrc[i] - histogramDist[current_dist + 1])) {
	 			result->emplace(make_pair(i, current_dist));
	 		 } else {
	 			result->emplace(make_pair(i, current_dist + 1));
	 		 }

	 		 ++i;

	 	 }

	 	 return result;
}

unique_ptr<vector<float>> getCumulativeHistogramFromImage(const Mat& image) {
	// not cumulative histogram
	unique_ptr<vector<int>> notNormalizedHistogram(getHistogramFromImage(image));
	unique_ptr<vector<float>> histogram = normalizeHist<int>(*notNormalizedHistogram);
	assert(abs(accumulate(histogram->begin(), histogram->end(), 0.0)  - 1) < pow(10, -6));

	assert(histogram->size() > 0);
	for (auto left = histogram->begin(), right = left + 1;
			right != histogram->end();
			++right, ++left) {
		*right += *left;
	}

	return normalizeHist<float>(*histogram);
}

void applyColorMapToImage(Mat& image, map<int, int>& colorMap) {
	for(int y = 0; y < image.rows; y++)
	{
	    for(int x = 0; x < image.cols; x++)
	    {
	    	unsigned char oldColor = image.at<uchar>(y,x);
	    	image.at<uchar>(y,x) =
	    			saturate_cast<unsigned char>(colorMap[oldColor]);

	    }

	}
}



unique_ptr<vector<int>> getHistogramFromImage(const Mat& image) {
	unique_ptr<vector<int>> histogram (new vector<int> (256));

	for (int r = 0; r < image.rows; r++) {
		for (int c = 0; c < image.cols; c++) {
			(*histogram)[image.at<unsigned char>(r, c)]++;
		}
	}

	return histogram;
}
