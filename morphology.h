

#ifndef MORPHOLOGY_H_
#define MORPHOLOGY_H_

#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>
#include <opencv2/core/saturate.hpp>

using namespace cv;

void binarize(const Mat& src, Mat& dist, int);

int find_mean_value(const Mat& src);
int find_treshold_otsu(const Mat& src);

#endif /* MORPHOLOGY_H_ */
