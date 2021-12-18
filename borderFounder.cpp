#include "borderFounder.h"

#define the_threshold 10

template<int matrix_size>
void calculateOperator(const Mat& src, Mat& result, const std::array<std::array<int, matrix_size>, matrix_size>& operator_matrix1,
		const std::array<std::array<int, matrix_size>, matrix_size>& operator_matrix2, int center) {

	result.setTo(Scalar(0));
	for (int column = 0; column < src.cols - matrix_size + 1; column++) {
		for (int row = 0; row < src.rows - matrix_size + 1; row++) {
			int response1 = 0;
			int response2 = 0;
			for (int i = 0; i < matrix_size; i++) {
				for (int j = 0; j < matrix_size; j++) {
					response1 +=  operator_matrix1[i][j] * src.at<uchar>(row + i, column + j);
					response2 +=  operator_matrix2[i][j] * src.at<uchar>(row + i, column + j);
				}
			}
			result.at<uchar>(row + center, column + center) =
					saturate_cast<unsigned char>((abs(response1) + abs(response2)) / (2 * matrix_size * matrix_size));
		}
	}
}


void apply_threshold(Mat& image, int threshold) {
	for (int column = 0; column < image.cols; ++column) {
		for (int row = 0; row < image.rows; ++row) {
			int color;
			if (image.at<uchar>(row, column) < threshold) {
				color = 0;
			} else {
				color = 240;
			}
			image.at<uchar>(row, column) = saturate_cast<unsigned char>(color);
		}
	}
}


void my_sobel(const Mat& src, Mat& dist) {
	const int operator_size = 3;

	std::array<std::array<int, operator_size>, operator_size> op1=
	          {{
			    {-1, -2, -1},
				{ 0,  0,  0},
				{ 1,  2,  1}
	          }};
	std::array<std::array<int, operator_size>, operator_size> op2 =
	          {{
			    {-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1}
	          }};
	calculateOperator<operator_size>(src, dist, op1, op2, 1);
	apply_threshold(dist, the_threshold);
}


void my_roberts(const Mat& src, Mat& dist) {
	const int operator_size = 2;

	std::array<std::array<int, operator_size>, operator_size> op1=
	          {{
			    {-1,  0},
				{ 0,  1},

	          }};
	std::array<std::array<int, operator_size>, operator_size> op2 =
	          {{
			    {0, -1},
				{1,  0},
	          }};
	calculateOperator<operator_size>(src, dist, op1, op2, 0);
	apply_threshold(dist, the_threshold);
}


void my_previt(const Mat& src, Mat& dist) {
	const int operator_size = 3;

	std::array<std::array<int, operator_size>, operator_size> op1=
	          {{
			    {-1, -1, -1},
				{ 0,  0,  0},
				{ 1,  1,  1}
	          }};
	std::array<std::array<int, operator_size>, operator_size> op2 =
	          {{
			    {-1, 0, 1},
				{-1, 0, 1},
				{-1, 0, 1}
	          }};
	calculateOperator<operator_size>(src, dist, op1, op2, 1);
	apply_threshold(dist, the_threshold);
}



