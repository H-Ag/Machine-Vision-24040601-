#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    Mat srcMat = imread("D:/data/HONOR Share/Honor Share/1.jpg");
    int height = srcMat.rows;
    int width = srcMat.cols;

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            // Vec3b顺序：B、G、R，每个通道像素值减半，图片变暗
            srcMat.at<Vec3b>(j, i)[0] = srcMat.at<Vec3b>(j, i)[0] / 2;
            srcMat.at<Vec3b>(j, i)[1] = srcMat.at<Vec3b>(j, i)[1] / 2;
            srcMat.at<Vec3b>(j, i)[2] = srcMat.at<Vec3b>(j, i)[2] / 2;
        }
    }

    imshow("test", srcMat);
    waitKey(0);
    return 0;
}
