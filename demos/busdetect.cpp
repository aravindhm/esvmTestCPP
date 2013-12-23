/*
 * Author: Ishan Misra
 * Copyright (C) 2012-13 by Ishan Misra
 * This file is part of the esvmTestCPP library.
 * It is released under the MIT License.
 * Project Homepage: https://github.com/imisra/esvmTestCPP
 * 
 * demo03.cpp : Using the testing pipeline for Exemplar-SVMs
 */
#include "esvm.h"
#include "esvm_utils.h"

#include <getopt.h>
#include <execinfo.h>
#include <signal.h>
#include <errno.h>
using namespace std;

class busParameters : public esvmParameters {
    public:
        cv::Rect roi;
        cv::Rect cat1;
        cv::Rect cat2;
        busParameters() : esvmParameters() {
          roi.x = 185; 
          roi.y = 15; 
          roi.width = 265;
          roi.height = 300;

          cat1 = roi;
          cat1.y = 15;
          cat1.height = 85;
          cat2 = roi;
          cat2.y = 60;
          cat2.height = 255;
        }
};

void handler(int sig) {
	void *array[100];
	size_t size;

	size = backtrace(array,100);

	fprintf(stderr,"Signal %d caught\n",sig);
	fprintf(stderr,"Signal Name: %s\n",strsignal(sig));
	fprintf(stderr,"Error Name: %s\n",strerror(errno));
	backtrace_symbols_fd(array,size,2);
	exit(1);
}


int main(int argc, char *argv[])
{
	
	const char *descFile = "../hybrid-models/exemplars/exemplar-txt-files-list";
	int numExemplars = 619;
	
	//install handler for debugging
	signal(SIGSEGV,handler);
	signal(SIGABRT,handler);

	//load the models
	esvmModel *model = loadExemplars(descFile,numExemplars);
	if(numExemplars != model->hogpyr->num) {
		fprintf(stderr,"could not load %d exemplars. Will work with %d exemplars\n",
				numExemplars,model->hogpyr->num);
		numExemplars = model->hogpyr->num;				
	}

	//get default parameters for classification
	busParameters params;
	params.detectionThreshold = -1.0;
	params.minImageScale = 1.0;
	
        //cv::VideoCapture cap("/dev/video1");
        //if(!cap.isOpened()) {
        //    return -1;
        //}
        int counter;
        char filename[256];
        for(counter = 1; counter < 4000; counter+=100)
        {
            sprintf(filename, "./JPEGImages/%08d.jpg", counter);
            printf("%s\n", filename);
            cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
            //cap >> img; // get a new frame from camera
            printf("Image read\n");

            //now extract category 1 and perform detection
            cv::Mat croppedImageCat1 = img(params.cat1).clone();
            printf("Crop1 done\n");
            esvmOutput *output1 = esvmSIMEWrapper(&params, croppedImageCat1, model);
            printf("crop1 detection done\n");
	    //adjust the output to undo the effect of the ROI
	    for(int j=0;j<output1->boxes->num;j++) {
                BOX_RMIN(output1->boxes, j) = BOX_RMIN(output1->boxes, j) + params.cat1.y;
                BOX_CMIN(output1->boxes, j) = BOX_CMIN(output1->boxes, j) + params.cat1.x;
                BOX_RMAX(output1->boxes, j) = BOX_RMAX(output1->boxes, j) + params.cat1.x;
                BOX_CMAX(output1->boxes, j) = BOX_CMAX(output1->boxes, j) + params.cat1.y;
            } 
            printf("crop1 antiroi updates done\n");
            //now extract category 2 and perform detection
            cv::Mat croppedImageCat2 = img(params.cat2).clone();
            printf("crop 2 done\n");
            esvmOutput *output2 = esvmSIMEWrapper(&params, croppedImageCat2, model);
            printf("crop 2 detection done\n");
	    //adjust the output to undo the effect of the ROI
	    for(int j=0;j<output2->boxes->num;j++) {
                BOX_RMIN(output2->boxes, j) = BOX_RMIN(output2->boxes, j) + params.cat2.y;
                BOX_CMIN(output2->boxes, j) = BOX_CMIN(output2->boxes, j) + params.cat2.x;
                BOX_RMAX(output2->boxes, j) = BOX_RMAX(output2->boxes, j) + params.cat2.y;
                BOX_CMAX(output2->boxes, j) = BOX_CMAX(output2->boxes, j) + params.cat2.x;
            } 
            printf("crop2 antiroi updates done\n");

            //combine output1 and output2
            esvmOutput *output = (esvmOutput *)esvmMalloc(sizeof(esvmOutput));
            output->boxes = (esvmBoxes*)esvmMalloc(sizeof(esvmBoxes));
            output->boxes->num = output1->boxes->num + output2->boxes->num;
            output->boxes->arr = (float*)esvmMalloc(output->boxes->num*ESVM_BOX_DIM*sizeof(float));
            for(int j = 0; j < output1->boxes->num; j++) {
                ARR_COPY(output1->boxes->arr, j, output->boxes->arr, j);
            }
            for(int j = 0; j < output2->boxes->num; j++) {
                ARR_COPY(output2->boxes->arr, j, output->boxes->arr, (j+output1->boxes->num));
            }

  	    // perform nms
            esvmOutput *final = (esvmOutput *)esvmMalloc(sizeof(esvmOutput));
            nms(output->boxes->arr, output->boxes->num, params.nmsOverlapThreshold, &(final->boxes->num), &(final->boxes->arr));

	    printf("Image %d; Weights %d; Tasks %d; Boxes %d;"
			    " Levels per octave %d; Hog Time %0.4lf; "
			    "Conv Time %0.4lf; NMS_etc Time %0.4lf\n",
			    counter, model->num,
			    params.userTasks,final->boxes->num,params.levelsPerOctave,
			    output1->perf.hogTime + output2->perf.hogTime,
                            output1->perf.convTime + output2->perf.convTime,
                            output1->perf.nmsTime + output2->perf.nmsTime);

	    printf("Total Boxes %d\n",final->boxes->num);

	    for(int j=0;j<final->boxes->num;j++) {
		    printf("bbox (%0.2f %0.2f %0.2f %0.2f); score %0.3f; scale %0.3f; exid %d; class %s\n",
				    BOX_RMIN((final->boxes),j), BOX_CMIN((final->boxes),j),
				    BOX_RMAX((final->boxes),j), BOX_CMAX((final->boxes),j),
				    BOX_SCORE((final->boxes),j), BOX_SCALE((final->boxes),j),
				    BOX_EXID((final->boxes),j),
				    model->idMap[BOX_CLASS((final->boxes),j)].c_str());


		    cv::rectangle(img,
				    cv::Point(BOX_CMIN((final->boxes),j),BOX_RMIN((final->boxes),j)),
				    cv::Point(BOX_CMAX((final->boxes),j),BOX_RMAX((final->boxes),j)),
				    cv::Scalar(0,0,255),
				    3,8,0
				 );
                  
	    }
            //cv::rectangle(img, cv::Point(params.roi.x, params.roi.y), cv::Point(params.roi.x+params.roi.width, params.roi.y+params.roi.height), cv::Scalar(0, 255, 0), 3, 8, 0);

	    //cv::namedWindow("hog-pipeline",CV_WINDOW_AUTOSIZE);
	    //cv::imshow("hog-pipeline",img);

	    //cv::waitKey(0);

	    //cv::destroyWindow("hog-pipeline");
            //cv::destroyWindow("hog-pipeline-debug");
	    cv::imwrite("detect-results.png",img);
	    printf("Saved detections as detect-results.png\n");
           
            free(output);
            free(output1);
            free(output2);
        }

	return 0;
}
