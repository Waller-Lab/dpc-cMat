
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
    for (int pIdx = 0; pIdx < stackCount; pIdx++)
    {
        char winTitle[100]; // Temporary title string
        sprintf(winTitle, "Image %d of %d", pIdx + 1, stackCount);
        showImg(imgStack[pIdx].getMat(ACCESS_READ), winTitle, -1);
    }
    cv::waitKey();
    cv::destroyAllWindows();
}

void showCMatStack(cvc::cMat * &imgStack, int stackCount, std::string title = "") {
    if (title == "") {
        title = "Image ";
    }
    for (int i = 0; i < stackCount; i++) {
        showImg(imgStack[i].real.getMat(ACCESS_READ), title + std::to_string(i)
            + ", Real Part", -1);
        showImg(imgStack[i].imag.getMat(ACCESS_READ), title + std::to_string(i)
            + ", Imaginary Part", -1);
        cvc::cMat ab = cvc::abs(imgStack[i]);
        showImg(ab.real.getMat(ACCESS_READ), title + std::to_string(i)
            + ", Norm", -1);
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

void oldMain(std::string jsonFileName, std::string imageFileName) {

    // Parse JSON file
    Json::Value calibrationJson;
    Json::Reader reader;
    ifstream jsonFile(jsonFileName);
    reader.parse(jsonFile, calibrationJson);

    double systemNA = calibrationJson.get("defaultNA",0).asFloat();
    double systemMag = calibrationJson.get("defaultMag",0).asFloat();

    // Parse calibration parameters
    double sourceCenterWidthHorz = calibrationJson.get("sourceCenterWidthHorz",0).asFloat();
    double sourceCenterWidthVert = calibrationJson.get("sourceCenterWidthVert",0).asFloat();
    //double lambda = calibrationJson.get("lambda",0).asFloat();
    double sourceRotation = calibrationJson.get("sourceRotation",0).asFloat();
    double ps = calibrationJson.get("pixelSize",0).asFloat();
    double ps_eff = ps/systemMag;

    //errors somewhere between here

	double lambda[3];
	lambda[0] = calibrationJson.get("lambda",0)[0].get("r",0).asDouble();
	lambda[1] = calibrationJson.get("lambda",0)[1].get("g",0).asDouble();
	lambda[2] = calibrationJson.get("lambda",0)[2].get("b",0).asDouble();
    std::cout << lambda[0] <<std::endl;
    std::cout << lambda[1] <<std::endl;
    std::cout << lambda[2] <<std::endl;

    //and here

    std::string sourceType = calibrationJson.get("sourceType","Quadrant").asString();
    int8_t nSources;
    // Get source calibration coefficients
    if (sourceType == "Quadrant")
        nSources = 4;
    else if (sourceType == "Tri")
        nSources = 3;
    std::cout<<"coeffs"<<std::endl;
    double sourceCoefficients[nSources][3];
    Json::Value sCoeffs = calibrationJson.get("sourceCoefficients",0);

    for (int16_t sIdx=0; sIdx<nSources; sIdx++)
    {
        sourceCoefficients[sIdx][0] = sCoeffs[sIdx][0].get("r",0).asDouble();
        sourceCoefficients[sIdx][1] = sCoeffs[sIdx][1].get("g",0).asDouble();
        sourceCoefficients[sIdx][2] = sCoeffs[sIdx][2].get("b",0).asDouble();

        std::cout << sourceCoefficients[sIdx][0]<<std::endl;
        std::cout << sourceCoefficients[sIdx][1]<<std::endl;
        std::cout << sourceCoefficients[sIdx][2]<<std::endl;
    }

	// Try to Load Image
	Mat img = imread(imageFileName, -1); // Load image
    if (img.rows ==0 || img.cols ==2)
    {
        std::cout << "ERROR - Image does not exist!" <<std::endl;
    }

	img.convertTo(img,CV_64FC1);
    //Mat imgC = Mat::zeros(img.rows/2,img.cols/2,CV_64FC3);
	Mat imgC[] = {Mat::zeros(img.rows/2,img.cols/2,CV_64FC1),
		          Mat::zeros(img.rows/2,img.cols/2,CV_64FC1),
				  Mat::zeros(img.rows/2,img.cols/2,CV_64FC1)};

	Raw2Color(img, imgC);

    showImg(imgC[0],"rawimageR", -1);
	showImg(imgC[1],"rawimageG", -1);
	showImg(imgC[2],"rawimageB", -1);
	showImgC(imgC, "rawimageRGB", COLORIMAGE_REAL);


    // Generate Sources
    cv::Mat Source[] = {cv::Mat::zeros(imgC[0].rows, imgC[0].cols, CV_64FC2),
                        cv::Mat::zeros(imgC[0].rows, imgC[0].cols, CV_64FC2),
                        cv::Mat::zeros(imgC[0].rows, imgC[0].cols, CV_64FC2)};

	SourceComputeOld(Source,sourceRotation, systemNA, lambda, ps_eff,
	              sourceCenterWidthHorz, sourceCenterWidthVert, sourceCoefficients);

	showImgC(Source, "Source", COLORIMAGE_COMPLEX);

	// Generate Transfer Functions
	Mat HiList[] = {cv::Mat::zeros(imgC[0].rows,imgC[0].cols,CV_64FC2),
		            cv::Mat::zeros(imgC[0].rows,imgC[0].cols,CV_64FC2),
					cv::Mat::zeros(imgC[0].rows,imgC[0].cols,CV_64FC2)};
    Mat HrList[] = {cv::Mat::zeros(imgC[0].rows,imgC[0].cols,CV_64FC2),
		            cv::Mat::zeros(imgC[0].rows,imgC[0].cols,CV_64FC2),
		     		cv::Mat::zeros(imgC[0].rows,imgC[0].cols,CV_64FC2)};

	for (int sIdx=0; sIdx<3; sIdx++)
	{
		// Generate Pupil
		Mat Pupil = Mat::zeros(imgC[0].rows,imgC[0].cols,CV_64FC2);
		PupilComputeOld(Pupil, systemNA, lambda[sIdx], ps_eff);

		//showComplexImg(Source[sIdx],SHOW_COMPLEX_REAL,"Source");

		HrHiComputeOld(HiList[sIdx], HrList[sIdx], Source[sIdx], Pupil, lambda[sIdx]);
		showComplexImg(HrList[sIdx],SHOW_COMPLEX_REAL,"Hr", cv::COLORMAP_JET);
		showComplexImg(HiList[sIdx],SHOW_COMPLEX_IMAGINARY,"Hi", cv::COLORMAP_JET);
	}

    // Deconvolution Process

	cv::Mat CDPC_Results = cv::Mat::zeros(imgC[0].rows,imgC[0].cols,CV_64FC2);
	std::complex<double> Regularization = std::complex<double>(1.0e-1,1.0e-3);

	cv::Mat A[5];

    GenerateAOld(A,HrList,HiList,Regularization);

	ColorDeconvolution_L2Old(CDPC_Results, imgC, A, HrList, HiList, lambda[1]);
	//writes the output matrix to a file for comparison
    /*
	std::ofstream output("outputMatrix.txt");
	output << "Rows: " << CDPC_Results.rows << std::endl;
	output << "Cols: " << CDPC_Results.cols << std::endl;
	for(int i = 0; i < CDPC_Results.rows; i++) // loop through y
 	{
    	const double* m_i = CDPC_Results.ptr<double>(i);
     	for(int j = 0; j < CDPC_Results.cols; j++)
     	{
     		output << m_i[CDPC_Results.cols * i + j] << " ";
     		if (j % 10 == 0) {
     			output << std::endl;
     		}
    	}
 	}
	output.close();
    */
	showComplexImg(CDPC_Results,SHOW_COMPLEX_REAL,"Recovered Amplitude",-1);
	showComplexImg(CDPC_Results,SHOW_COMPLEX_IMAGINARY,"Recovered Phase", cv::COLORMAP_JET);

}

void testMain() {
    std::string json = "testDataset_dpc.json";

    //Parse json
    Json::Value calibrationJson;
    Json::Reader reader;
    ifstream jsonFile (json);
    //reads jsonFile into calibrationJson
    reader.parse(jsonFile, calibrationJson);

    //get component objects
    Json::Value common = calibrationJson.get("common", 0);
    Json::Value dpc = calibrationJson.get("dpc", 0);

    if (common == 0 || dpc == 0) {
        std::cout << "ERROR: SOME PARAMETERS NOT PROVIDED" << std::endl;
        return;
    }

    std::string imageFileName = common.get("imgStackFileName", 0).asString();
    std::cout << "Image File Name: " << imageFileName << std::endl;

    std::string type = common.get("datasetType", 0).asString();

    UMat* mats;
    uint16_t nImgs;

    if (type == "dpc") {
        nImgs = loadImageStack(imageFileName.c_str(), mats);
//        mats.convertTo(mats, CV_64FC1);
        for (int i = 0; i < nImgs; i++) {
            mats[i].convertTo(mats[i], CV_64FC1);
        }
    } else if (type == "cdpc") {
        mats = new UMat[3];
        nImgs = 3;
        //load image
        Mat img = imread(imageFileName, -1);
        if (img.rows ==0 || img.cols ==2)
        {
            std::cout << "ERROR - Image does not exist!" <<std::endl;
        }

        img.convertTo(img,CV_64FC1);
        //Mat imgC = Mat::zeros(img.rows/2,img.cols/2,CV_64FC3);
        Mat imgC[] = {Mat::zeros(img.rows/2,img.cols/2,CV_64FC1),
                      Mat::zeros(img.rows/2,img.cols/2,CV_64FC1),
                      Mat::zeros(img.rows/2,img.cols/2,CV_64FC1)};
        Raw2Color(img, imgC);
        for (int i = 0; i < 3; i++) {
            mats[i] = imgC[i].getUMat(cv::ACCESS_RW);
        }
    } else {
        std::cout << "ERROR: Must give a type" << std::endl;
        return;
    }

    cv::Size size = mats[0].size();
    cvc::cMat* images = new cvc::cMat[nImgs];
    for (int i = 0; i < nImgs; i++) {
        images[i] = *(new cvc::cMat(mats[i]));
    }
    showCMatStack(images, nImgs, "Image ");

    double systemNa = common.get("objectiveNa", 0).asFloat();      //TODO systemNa = objectiveNa?
    double illumNa = common.get("illuminationNa", 0).asFloat();
    double systemMag = common.get("systemMag", 0).asFloat();

    std::cout << "System NA is: " << systemNa << std::endl;
    std::cout << "Illumination NA is: " << illumNa << std::endl;
    std::cout << "System Magnification is: " << systemMag << std::endl;
    std::cout << "Image size is: " << size << std::endl;

    double sourceCenterWidthHorz = common.get("sourceCenterWidthHorz", 0).asFloat();
    double sourceCenterWidthVert = common.get("sourceCenterWidthVert", 0).asFloat();

    std::cout << "Source Center Horizontal: " << sourceCenterWidthHorz << std::endl;
    std::cout << "Source Center Vertical: " << sourceCenterWidthVert << std::endl;

    Json::Value sourceRotationLst = dpc.get("sourceRotationList", 0);
    double RotAngle = dpc.get("sourceRotation", 0).asFloat();       //TODO 0 or 90?

    //does this do anything?
    double srcRot [sourceRotationLst.size()];
    for (int i = 0; i < sourceRotationLst.size(); i++) {
        srcRot[i] = sourceRotationLst[i].asFloat();
        std::cout << "Source Rotation #" << i << "\n" << srcRot[i] << std::endl;
    }
    std::cout << "Source Rotation Angle: " << srcRot << std::endl;

    //access values by sourceRotation[i].asFloat()
//    std::cout << rotVal[1].asFloat() << std::endl;
    double ps = common.get("pixelSize",0).asFloat();
    double ps_eff = ps/systemMag;

    std::cout << "Pixel Size is: " << ps << std::endl;

    Json::Value lambdas = common.get("wavelengthList", 0);
//    double lambda [3];
    double lambda [4];

    lambda[0] = lambdas[0].asDouble();
    lambda[1] = lambdas[1].asDouble();
    lambda[2] = lambdas[2].asDouble();
    lambda[3] = lambdas[3].asDouble();
    std::cout << "Wavelength 1 is: " << lambda[0] << std::endl;
    std::cout << "Wavelength 2 is: " << lambda[1] << std::endl;
    std::cout << "Wavelength 3 is: " << lambda[2] << std::endl;
    std::cout << "Wavelength 4 is: " << lambda[3] << std::endl;

    std::string sourceType = common.get("sourceType","Quadrant").asString();
    int8_t nSources;
    // Get source calibration coefficients
    // TODO determine if we want Quadrant or Tri
    if (sourceType == "Quadrant") {
        nSources = 4;
    } else if (sourceType == "Tri") {
        nSources = 3;
    }
    //TODO use nSources or nImgs for counting to 4?
    // Get source calibration coefficients. Read down the columns
    std::cout<< "coeffs" <<std::endl;
    double sourceCoefficients[nSources][4];     //TODO make it [4] sized probs
    Json::Value sCoeffs = dpc.get("sourceCoefficients",0);
    for (int16_t sIdx=0; sIdx<nSources; sIdx++) {

        Json::Value temp = sCoeffs[sIdx];
//        std::cout << temp.isArray() << std::endl;
        // sourceCoefficients[sIdx][0] = temp[0].asDouble();
        // sourceCoefficients[sIdx][1] = temp[1].asDouble();
        // sourceCoefficients[sIdx][2] = temp[2].asDouble();
        // sourceCoefficients[sIdx][3] = temp[3].asDouble();

        sourceCoefficients[0][sIdx] = temp[0].asDouble();
        sourceCoefficients[1][sIdx] = temp[1].asDouble();
        sourceCoefficients[2][sIdx] = temp[2].asDouble();
        sourceCoefficients[3][sIdx] = temp[3].asDouble();

//        sourceCoefficients[sIdx][0] = sCoeffs[sIdx][0].get("r",0).asDouble();
//        sourceCoefficients[sIdx][1] = sCoeffs[sIdx][1].get("g",0).asDouble();
//        sourceCoefficients[sIdx][2] = sCoeffs[sIdx][2].get("b",0).asDouble();

        // std::cout << sourceCoefficients[sIdx][0] << std::endl;
        // std::cout << sourceCoefficients[sIdx][1] << std::endl;
        // std::cout << sourceCoefficients[sIdx][2] << std::endl;
        // std::cout << sourceCoefficients[sIdx][3] << std::endl;
    }

    for (int sIdx = 0; sIdx < nSources; sIdx++) {
        for (int i = 0; i < 4; i++) {
            std::cout << sourceCoefficients[sIdx][i] << " ";
        }
        std::cout << std::endl;
    }

    // for each source (trial number) index first. Ie to get quadrants for first
    // trial use sourceCoefficients[sIdx][i], where i is 1 if you want that quadrant
    // on and 0 if you want that quadrant off

//     //load image
//     Mat img = imread(imageFileName, -1);
//     if (img.rows ==0 || img.cols ==2)
//     {
//         std::cout << "ERROR - Image does not exist!" <<std::endl;
//     }
//
//     img.convertTo(img,CV_64FC1);
//     //Mat imgC = Mat::zeros(img.rows/2,img.cols/2,CV_64FC3);
//     Mat imgC[] = {Mat::zeros(img.rows/2,img.cols/2,CV_64FC1),
//                   Mat::zeros(img.rows/2,img.cols/2,CV_64FC1),
//                   Mat::zeros(img.rows/2,img.cols/2,CV_64FC1)};
//
// //    Raw2Color(img, imgC);

    // cv::UMat imgs [3];
    // for (int i = 0; i < 3; i++) {
    //     imgs[i] = imgC[i].getUMat(cv::ACCESS_RW);
    // }
    // cv::UMat * imPtr = imgs;
    // //TODO why not showing images? it works for one with a Mat
    // showImgStack(imPtr, 3);

    // Generate Sources
    // cvc::cMat Source[] = {
    //         cvc::zeros(imgC[0].rows, imgC[0].cols),
    //         cvc::zeros(imgC[0].rows, imgC[0].cols),
    //         cvc::zeros(imgC[0].rows, imgC[0].cols)
    // };
    //TODO might have to add a fourth source
    cvc::cMat Source[] = {
            cvc::zeros(size),
            cvc::zeros(size),
            cvc::zeros(size),
            cvc::zeros(size)
    };

    SourceCompute(Source, RotAngle, systemNa, lambda, ps_eff,
        sourceCenterWidthHorz, sourceCenterWidthVert, sourceCoefficients);
    cvc::cMat* ptr;
    ptr = Source;
    showCMatStack(ptr, 4, "Source ");

	// Generate Transfer Functions

    cvc::cMat HaList[] = {
        cvc::zeros(size),
        cvc::zeros(size),
        cvc::zeros(size),
        cvc::zeros(size)
    };

    cvc::cMat HpList[] = {
        cvc::zeros(size),
        cvc::zeros(size),
        cvc::zeros(size),
        cvc::zeros(size)
    };

    for (int sIdx=0; sIdx<nSources; sIdx++) {
    	// Generate Pupil
    	cvc::cMat Pupil = cvc::zeros(size);
    	PupilCompute(Pupil, systemNa, lambda[sIdx], ps_eff);
        // ptr = &Pupil;
        // showCMatStack(ptr, 1);

        //HrHiCompute(HrList[sIdx], HiList[sIdx], Source[sIdx], Pupil, lambda[sIdx]);
        HaHpCompute(HaList[sIdx], HpList[sIdx], Source[sIdx], Pupil, lambda[sIdx]);
	}
    ptr = HaList;
//    showCMatStack(ptr, 4, "Amplitude Transfer Function ");

    ptr = HpList;
//    showCMatStack(ptr, 4, "Phase Transfer Function");

    // Deconvolution Process

	cvc::cMat CDPC_Results = cvc::zeros(size);
	std::complex<double> Regularization = std::complex<double>(1.0e-1,1.0e-3);

	cvc::cMat A[5];

    //TODO breaks inside GenerateA-- find where
    GenerateA(A, HaList, HpList, Regularization);

//	ColorDeconvolution_L2(CDPC_Results, imgC, A, HrList, HiList, lambda[1]);
    ColorDeconvolution_L2(CDPC_Results, images, A, HaList, HpList, lambda[1]);
//	showComplexImg(CDPC_Results,SHOW_COMPLEX_REAL,"Recovered Amplitude",-1);
//	showComplexImg(CDPC_Results,SHOW_COMPLEX_IMAGINARY,"Recovered Phase", cv::COLORMAP_JET);
    showImg(CDPC_Results.real.getMat(ACCESS_READ), "Recovered Amplitude", -1);
    showImg(CDPC_Results.imag.getMat(ACCESS_READ), "Recovered Phase");

}

void runTests() {
//    testRange();
//    oldMain("testDataset_dpc.json", "testDataset_dpc.tif");
    testMain();
}

int main(int argc, char** argv)
{
    /*
    const char * datasetFilename_dpc_tif = "./testDataset_dpc.tif";
    cv::UMat * imgStack;
    uint16_t imgCount = loadImageStack(datasetFilename_dpc_tif, imgStack);
    showImgStack(imgStack, imgCount);
    */
    //runTests();
    testMain();
}
