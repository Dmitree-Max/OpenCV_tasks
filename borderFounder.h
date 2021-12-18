

#ifndef BORDERFOUNDER_H_
#define BORDERFOUNDER_H_

#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>
#include <opencv2/core/saturate.hpp>

using namespace std;
using namespace cv;

template<int matrix_size>
void calculateOperator(const Mat& src, Mat& result, const std::array<std::array<int, matrix_size>, matrix_size>& operator_matrix1,
		const std::array<std::array<int, matrix_size>, matrix_size>& operator_matrix2, int center);
void apply_threshold(Mat& image, int threshold);
void my_sobel(const Mat& src, Mat& dist);
void my_previt(const Mat& src, Mat& dist);
void my_roberts(const Mat& src, Mat& dist);

#endif /* BORDERFOUNDER_H_ */
