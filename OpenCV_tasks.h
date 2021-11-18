
#ifndef OPENCV_TASKS_H_
#define OPENCV_TASKS_H_

#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>
#include <opencv2/core/saturate.hpp>

using namespace cv;
using namespace std;

void linearyChangeImage(const Mat& input, Mat& output, float coeffecient);
unique_ptr<vector<int>> getHistogramFromImage(const Mat& image);
void applyColorMapToImage(Mat& image, map<int, int>& colorMap);
unique_ptr<vector<float>> getCumulativeHistogramFromImage(const Mat& image);
unique_ptr<map<int, int>> getColorMap(const vector<float>& histogramDist,
			const vector<float>& histogramSrc);
void histogramApplication(const Mat& input, const Mat& dist, Mat& output);

template <typename T>
unique_ptr<vector<float> > normalizeHist(const vector<T>& histogram);

void equalizeImage(const Mat& input, Mat& output);

#endif /* OPENCV_TASKS_H_ */
