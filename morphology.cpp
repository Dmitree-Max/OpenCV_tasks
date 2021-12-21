
#include "morphology.h"

using namespace cv;

void binarize(const Mat& src, Mat& dist) {
	dist = src.clone();
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			dist.at<uchar>(r, c) = src.at<uchar>(r, c) < 128 ? 0 : 240;
		}
	}
}


void dilatation(const Mat& src, Mat& dist, std::vector<std::vector<uchar>>& mask, std::pair<int, int> center) {
	dist = src.clone();
	assert(mask[center.first][center.second] == 0
			or mask[center.first][center.second] == 240);
	const uchar center_color = mask[center.first][center.second];
	int mask_h = mask.size();
	int mask_w = mask[0].size();
	for(int row = 0; row < src.rows; row++) {
		for(int col = 0; col < src.cols; col++) {
			if (src.at<uchar>(row, col) == center_color) {
				for (int mask_row = 0; mask_row < mask_h; mask_row++) {
					for (int mask_col = 0; mask_col < mask_w; mask_col++) {
						if (row - (center.first - mask_row) > 0 and col - (center.second - mask_col) > 0
								and row - (center.first - mask_row) < src.rows
								and col - (center.second - mask_col) < src.cols) {
							dist.at<uchar>(row - (center.first - mask_row), col - (center.second - mask_col)) = center_color;
						}
					}
				}
			}
		}
	}
}



void erosion(const Mat& src, Mat& dist, std::vector<std::vector<uchar>>& mask, std::pair<int, int> center) {
	dist = src.clone();
	assert(mask[center.first][center.second] == 0
			or mask[center.first][center.second] == 240);
	const uchar center_color = mask[center.first][center.second];
	const uchar oposite_center_color = 240 - center_color;
	int mask_h = mask.size();
	int mask_w = mask[0].size();
	for(int row = 0; row < src.rows; row++) {
		for(int col = 0; col < src.cols; col++) {
			int center_pixel_erozed = false;
			for (int mask_row = 0; mask_row < mask_h; mask_row++) {
				for (int mask_col = 0; mask_col < mask_w; mask_col++) {
					if (row - (center.first - mask_row) > 0 and col - (center.second - mask_col) > 0
							and row - (center.first - mask_row) < src.rows
							and col - (center.second - mask_col) < src.cols) {
						if (src.at<uchar>(row - (center.first - mask_row), col - (center.second - mask_col))
								!= mask[mask_row][mask_col]) {
							center_pixel_erozed = true;
							break;
						}
					}
				}
			}

			if (center_pixel_erozed) {
				dist.at<uchar>(row, col) = oposite_center_color;
			}
		}
	}
}



void open_operation(const Mat& src, Mat& dist, std::vector<std::vector<uchar>>& mask, std::pair<int, int> center) {
	Mat temp;
	erosion(src, temp, mask, center);
	dilatation(temp, dist, mask, center);
}


void close_operation(const Mat& src, Mat& dist, std::vector<std::vector<uchar>>& mask, std::pair<int, int> center) {
	Mat temp;
	dilatation(src, temp, mask, center);
	erosion(temp, dist, mask, center);
}
