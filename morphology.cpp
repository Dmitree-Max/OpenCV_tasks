
#include "morphology.h"
#include <memory>
#include "histogramManipulation.h"

using namespace cv;
using namespace std;

void binarize(const Mat& src, Mat& dist, int treshold) {
	dist = src.clone();
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			dist.at<uchar>(r, c) = src.at<uchar>(r, c) < treshold ? 0 : 240;
		}
	}
}



int find_mean_value(const Mat& src) {
	unique_ptr<vector<int>> hist = getHistogramFromImage(src);

	int left = 0;
	int right = src.cols * src.rows;

	auto it = hist->begin();
	for (; it != hist->end(); ++it) {
		left += *it;
		right -= *it;
		if (left >= right) {
			break;
		}
	}

	return distance(hist->begin(), it);
}


static int calc_intencity_sum(const vector<int>& hist) {
	int result = 0;

	int i = 0;
	for (auto it = hist.begin(); it != hist.end(); ++it, ++i) {
		result += *it * i;
	}

	return result;
}


int find_treshold_otsu(const Mat& src) {
	unique_ptr<vector<int>> hist = getHistogramFromImage(src);
	int left_pixel_count = 0;
	int right_pixel_count = src.cols * src.rows;
	int left_intencity_sum = 0;
	int right_intencity_sum = calc_intencity_sum(*hist);

	int best_treshold = 0;
	double best_sigma = DBL_MIN;
	for (int i = 0; i < 256; ++i) {
		left_pixel_count += (*hist)[i];
		right_pixel_count -= (*hist)[i];
		left_intencity_sum += (*hist)[i] * i;
		right_intencity_sum -= (*hist)[i] * i;

		double left_class_prob = left_pixel_count / (double) (src.cols * src.rows);
		double right_class_prob = 1.0 - left_class_prob;

		double left_class_mean = left_intencity_sum / (double) left_pixel_count;
		double right_class_mean = right_intencity_sum / (double) right_pixel_count;

		double mean_delta = right_class_mean - left_class_mean;

		double sigma = left_class_prob * right_class_prob * mean_delta * mean_delta;

		if (sigma > best_sigma) {
			best_treshold = i;
			best_sigma = sigma;
		}
	}

	return best_treshold;
}



















