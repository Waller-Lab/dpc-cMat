
#include <stdio.h>
#include <cmath>
#include <iostream>

#include "../cMat/cMat.cpp"
//#include "../cMat/cMat.h"

#ifndef DPC_H
#define DPC_H 1

using namespace cvc;

namespace dpc {

    cMat range(double start, double end, double interval);

    //function from the old code
    void SourceComputeOld(cv::Mat& sourceOutput, double RotAngle, double NA_illum, double Lambda, double ps, double crossOffsetH, double crossOffsetV, const double quadrantCoefficents[][3]);
    void PupilComputeOld(cv::Mat& output, double NA, double Lambda, double ps);
    void Raw2Color(const cv::Mat& IntensityC, cv::Mat output);
    void showImgC(cv::Mat ImgC, std::string windowTitle);
    void HrHiComputeOld(cv::Mat& Hi, cv::Mat& Hr, cv::Mat& Source, cv::Mat& Pupil , double Lambda);

    //new functions with cMat
    void Normalize(cvc::cMat& output, cvc::cMat& input);
    void PupilCompute(cvc::cMat& output, double NA, double Lambda, double ps);
    void SourceCompute(cvc::cMat* sourceRgb, double RotAngle, double NA_illum, const double Lambda[4], double ps,
        double crossOffsetH, double crossOffsetV, const double quadrantCoefficents[][4]);
    void HaHpCompute(cvc::cMat& Ha, cvc::cMat& Hp, cvc::cMat& Source, cvc::cMat& Pupil , double Lambda);
    void GenerateA(cvc::cMat* output, cvc::cMat* HaList, cvc::cMat* HpList, std::complex<double> Regularization);
    void ColorDeconvolution_L2(cvc::cMat& output, cvc::cMat* Intensity, cvc::cMat* A, cvc::cMat* HaList,
        cvc::cMat* HpList, double Lambda);

    inline void drawRotatedRectMat(cv::Mat& image, cv::RotatedRect rRect, cv::Scalar color ) {

        cv::Point2f vertices2f[4];
        cv::Point vertices[4];
        rRect.points(vertices2f);
        for(int i = 0; i < 4; ++i){
            vertices[i] = vertices2f[i];
        }
        cv::fillConvexPoly(image, vertices,4,color);
    }

    inline void drawRotatedRect(cvc::cMat& image, cv::RotatedRect rRect, cv::Scalar color) {

        cv::Mat real = cv::Mat::zeros(image.size(), image.type());
        dpc::drawRotatedRectMat(real, rRect, color);
        image.real = real.getUMat(cv::ACCESS_RW);
    }

}
#endif
