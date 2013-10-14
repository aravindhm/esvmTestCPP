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
        busParameters() : esvmParameters() {
          roi.x = 185; 
          roi.y = 15; 
          roi.width = 165;
          roi.height = 300;
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

            //crop the image to look at the roi 
            cv::Mat croppedImage = img(params.roi).clone();

            esvmOutput *output = esvmSIMEWrapper(&params, croppedImage, model);
	
	    //adjust the output to undo the effect of the ROI
	    for(int j=0;j<output->boxes->num;j++) {
                BOX_RMIN(output->boxes, j) = BOX_RMIN(output->boxes, j) + params.roi.y;
                BOX_CMIN(output->boxes, j) = BOX_CMIN(output->boxes, j) + params.roi.x;
                BOX_RMAX(output->boxes, j) = BOX_RMAX(output->boxes, j) + params.roi.y;
                BOX_CMAX(output->boxes, j) = BOX_CMAX(output->boxes, j) + params.roi.x;
            } 

	    printf("Image %d; Hog levels %d; Weights %d; Tasks %d; Boxes %d;"
			    " Levels per octave %d; Hog Time %0.4lf; "
			    "Conv Time %0.4lf; NMS_etc Time %0.4lf\n",
			    counter,output->hogpyr->num,model->num,
			    params.userTasks,output->boxes->num,params.levelsPerOctave,
			    output->perf.hogTime,output->perf.convTime,output->perf.nmsTime);

	    printf("Total Boxes %d\n",output->boxes->num);

	    for(int j=0;j<output->boxes->num;j++) {
		    printf("bbox (%0.2f %0.2f %0.2f %0.2f); score %0.3f; scale %0.3f; exid %d; class %s\n",
				    BOX_RMIN((output->boxes),j), BOX_CMIN((output->boxes),j),
				    BOX_RMAX((output->boxes),j), BOX_CMAX((output->boxes),j),
				    BOX_SCORE((output->boxes),j), BOX_SCALE((output->boxes),j),
				    BOX_EXID((output->boxes),j),
				    model->idMap[BOX_CLASS((output->boxes),j)].c_str());


		   /* cv::rectangle(img,
				    cv::Point(BOX_CMIN((output->boxes),j),BOX_RMIN((output->boxes),j)),
				    cv::Point(BOX_CMAX((output->boxes),j),BOX_RMAX((output->boxes),j)),
				    cv::Scalar(0,0,255),
				    3,8,0
				 );
                   */
	    }

	    /*cv::namedWindow("hog-pipeline",CV_WINDOW_AUTOSIZE);
	    cv::imshow("hog-pipeline",img);
	    cv::waitKey(0);
	    cv::destroyWindow("hog-pipeline");
	    cv::imwrite("detect-results.png",img);
	    printf("Saved detections as detect-results.png\n");*/
           
        }

	return 0;
}
