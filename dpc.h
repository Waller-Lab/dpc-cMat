
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
    cMat genSrcAngular(cMat srcCoeffs, double rotationAngle, std::vector<int> imSize, double na, double ps, std::vector<double> lambdaList);
    std::vector<cMat> genHuHp(cMat src, cMat pupil, cMat lambdaList, bool shiftedOutput);
    cMat genMeasurementsLinear(cMat field, cMat pupil, cMat Hu, cMat Hp, std::vector<double> lambdaList);
    cMat genMeasurementsNonlinear(cMat field, cMat src, cMat pupil, double nPositions);
    cMat cdpcLeastSquares(cMat intensity, cMat Hu, cMat Hp, double ampReg, double phaseReg, double resultLambda);

    void SourceComputeOld(cv::Mat& sourceOutput, double RotAngle, double NA_illum, double Lambda, double ps, double crossOffsetH, double crossOffsetV, const double quadrantCoefficents[][3]);
    void PupilComputeOld(cv::Mat& output, double NA, double Lambda, double ps);
    void Raw2Color(const cv::Mat& IntensityC, cv::Mat output);
    void showImgC(cv::Mat ImgC, std::string windowTitle);
    void HrHiComputeOld(cv::Mat& Hi, cv::Mat& Hr, cv::Mat& Source, cv::Mat& Pupil , double Lambda);

    inline void drawRotatedRectOld(cv::Mat& image, cv::RotatedRect rRect, cv::Scalar color ) {

        cv::Point2f vertices2f[4];
        cv::Point vertices[4];
        rRect.points(vertices2f);
        for(int i = 0; i < 4; ++i){
            vertices[i] = vertices2f[i];
        }
        cv::fillConvexPoly(image, vertices,4,color);
    }

    inline void drawRotatedRect(cvc::cMat& image, cv::RotatedRect rRect, cv::Scalar color ) {

        cv::Point2f vertices2f[4];
        cv::Point vertices[4];
        rRect.points(vertices2f);
        for(int i = 0; i < 4; ++i){
            vertices[i] = vertices2f[i];
        }
        cvc::fillConvexPoly(image, vertices,4,color);
    }

}
#endif
