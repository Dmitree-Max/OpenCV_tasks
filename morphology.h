

#ifndef MORPHOLOGY_H_
#define MORPHOLOGY_H_

#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>
#include <opencv2/core/saturate.hpp>

using namespace cv;

void binarize(const Mat& src, Mat& dist);

void dilatation(const Mat& src, Mat& dist, std::vector<std::vector<uchar>>& mask, std::pair<int, int> center_position);
void erosion(const Mat& src, Mat& dist, std::vector<std::vector<uchar>>& mask, std::pair<int, int> center);

void open_operation(const Mat& src, Mat& dist, std::vector<std::vector<uchar>>& mask, std::pair<int, int> center);
void close_operation(const Mat& src, Mat& dist, std::vector<std::vector<uchar>>& mask, std::pair<int, int> center);

#endif /* MORPHOLOGY_H_ */
