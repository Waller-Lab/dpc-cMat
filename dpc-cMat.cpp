
#include <stdio.h>
#include <cmath>
#include <iostream>

#include "include/json.h"
//#include "cMat.h"
#include "./dpc.cpp"
//#include "./dpc.cpp"

#define libDebug 1

using namespace std;

namespace libtiff {
    #include "tiffio.h"
}

uint16_t loadImageStack(const char * fileName, cv::UMat * &imgStackToFill)
{
    libtiff::TIFF* tif = libtiff::TIFFOpen(fileName, "r");
    uint16_t pageCount = 0;
    if (tif) {

        pageCount += 1;
        uint16_t pageIdx = 0;

        // Get total number of frames in image
        while (libtiff::TIFFReadDirectory(tif))
            pageCount++;

        // Reset directiry index to zero
        libtiff::TIFFSetDirectory(tif,0);

        // Get metadata
        libtiff::uint32 W, H, imgLen;
        libtiff::uint16 Bands, bps, sf, photo, pn1, pn2, pConfig;

        libtiff::TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &W);
        libtiff::TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &H);
        libtiff::TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &Bands);
        libtiff::TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
        libtiff::TIFFGetField(tif,TIFFTAG_SAMPLEFORMAT, &sf);
        libtiff::TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);
        libtiff::TIFFGetField(tif, TIFFTAG_PAGENUMBER, &pn1, &pn2);
        libtiff::TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgLen);
        libtiff::TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &pConfig);

        if (libDebug)
        {
            std::cout << "loadImageStack: TIFF Properties:" << std::endl;
            std::cout << "  Page Count: " << pageCount << std::endl;
            std::cout << "  Width: " << W << std::endl;
            std::cout << "  Height: " << H << std::endl;
            std::cout << "  Bands: " << Bands << std::endl;
            std::cout << "  BitsPerSample: " << bps << std::endl;
            std::cout << "  SampleFormat: " << sf << std::endl;
            std::cout << "  Photometric: " << photo << std::endl;
            std::cout << "  Page Number 1: " << pn1 << std::endl;
            std::cout << "  Page Number 2: " << pn2 << std::endl;
            std::cout << "  Image Length: " << imgLen << std::endl;
            std::cout << "  Planar Config: " << pConfig << std::endl;
            std::cout << std::endl;
        }

        // Allocate array pointers for imgStack
        imgStackToFill = new cv::UMat[pageCount];

        libtiff::tsize_t ScanlineSizeBytes = libtiff::TIFFScanlineSize(tif);
        ushort * scanline;
    	libtiff::tdata_t buf;
    	libtiff::uint32 row;

        do {
            // Allocate memory for this page
            imgStackToFill[pageIdx] = cv::UMat::zeros(H, W, CV_16UC1);

            // Add a reference count to the page to prevent it from being deallocated
            imgStackToFill[pageIdx].addref();
            imgStackToFill[pageIdx].addref();

            // Get buffer of tiff data
        	buf = libtiff::_TIFFmalloc(libtiff::TIFFScanlineSize(tif));

            // Loop over rows of image and assign to cv::Mat
            for (row = 0; row < imgLen; row++)
            {
                libtiff::TIFFReadScanline(tif, buf, row);
                scanline = imgStackToFill[pageIdx].getMat(cv::ACCESS_RW).ptr<ushort>(row);
                memcpy(scanline, buf, ScanlineSizeBytes);
            }

            // Free buffer
            libtiff::_TIFFfree(buf);

            // Incriment page number
            pageIdx++;

        }while(libtiff::TIFFReadDirectory(tif));

        /*
        if (libDebug)
            showImgStack(imgStackToFill, pageCount);
        */

        libtiff::TIFFClose(tif);
    }
    else
    {
        imgStackToFill = NULL;
        std::cout << "Warning: TIFF file " << fileName <<" was not found!" <<std::endl;
    }
    return pageCount;

}

void showImgStack(cv::UMat * &imgStack, int stackCount)
{
    // Display Image Stack
    cv::UMat displayMat;

    for (int pIdx = 0; pIdx < stackCount; pIdx++)
    {
        imgStack[pIdx].convertTo(displayMat, CV_8UC1, 255.0 / (65536.0));
        cv::startWindowThread();
        char winTitle[100]; // Temporary title string
        sprintf(winTitle, "Image %d of %d", pIdx + 1, stackCount);
        cv::namedWindow(winTitle, cv::WINDOW_NORMAL);
        cv::imshow(winTitle, displayMat);
    }
    cv::waitKey();
    cv::destroyAllWindows();

}

void testRange() {
    cMat rng = dpc::range(0.0, 360.0, 60.0);
    std::cout << "Range to 360 is: " << rng << std::endl;
    cMat rng2 = dpc::range(400, 0, 20);
    std::cout << "Range from 400 to 0 w/ step size 20 is:" << rng2 << std::endl;
}

void oldRotatedRect() {
    cv::Point2f center (300, 500);
    cv::Size2f size (200, 400);
    //2 x 4 rectangle centered at 3, 3?
    cv::RotatedRect rect (center, size, 0.0);
    cv::Point2f pts [4];
    rect.points(pts);
    for (int i = 0; i < 4; i++) {
        std::cout << pts[i] << std::endl;
    }
    cv::Mat img = cv::Mat::zeros(1024, 1024, CV_64F);
    dpc::drawRotatedRectOld(img, rect, cv::Scalar(255, 0, 0));
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Rectangle", img);
    cv::waitKey(0);
}

void newRotatedRect() {
    cv::Point2f center (300, 500);
    cv::Size2f size (200, 400);
    //2 x 4 rectangle centered at 3, 3?
    cv::RotatedRect rect (center, size, 0.0);
    cv::Point2f pts [4];
    rect.points(pts);
    for (int i = 0; i < 4; i++) {
        std::cout << pts[i] << std::endl;
    }
    cvc::cMat img = cvc::zeros(1024, 1024);
    dpc::drawRotatedRect(img, rect, cv::Scalar(255, 0, 0));
    std::cout << "going into cmshow" << std::endl;
//    std::cout << "cmat is: " << img << std::endl;
    cvc::cmshow(img, "Rectangle");
}

void testPupilComputeOld() {
    
}

void testPupilComputeNew() {

}

void runTests() {
//    testRange();
//    oldRotatedRect();
    newRotatedRect();
//    testPupilComputeOld();
//    testPupilComputeNew();
}

int main(int argc, char** argv)
{
    /*
    const char * datasetFilename_dpc_tif = "./testDataset_dpc.tif";
    cv::UMat * imgStack;
    uint16_t imgCount = loadImageStack(datasetFilename_dpc_tif, imgStack);
    showImgStack(imgStack, imgCount);
    */
    runTests();
}
