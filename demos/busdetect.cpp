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
	
	const char *imageName = "../sample-data/aeroplane.jpg";
	const char *descFile = "../sample-data/exemplars/exemplar-txt-files-list";
	int numExemplars = 20;
	
	//this image is loaded only for display purposes. SIMEWrapper loads its own image.
	cv::Mat img = cv::imread(imageName,CV_LOAD_IMAGE_COLOR);
	

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
	esvmParameters *params = esvmDefaultParameters();
	params->detectionThreshold = -1.0;
	
	esvmOutput *output = esvmSIMEWrapper(params, img, model);
	
	
	printf("Image %s; Hog levels %d; Weights %d; Tasks %d; Boxes %d;"
			" Levels per octave %d; Hog Time %0.4lf; "
			"Conv Time %0.4lf; NMS_etc Time %0.4lf\n",
			imageName,output->hogpyr->num,model->num,
			params->userTasks,output->boxes->num,params->levelsPerOctave,
			output->perf.hogTime,output->perf.convTime,output->perf.nmsTime);
	
	printf("Total Boxes %d\n",output->boxes->num);

	for(int j=0;j<output->boxes->num;j++) {
		printf("bbox (%0.2f %0.2f %0.2f %0.2f); score %0.3f; scale %0.3f; exid %d; class %s\n",
				BOX_RMIN((output->boxes),j), BOX_CMIN((output->boxes),j),
				BOX_RMAX((output->boxes),j), BOX_CMAX((output->boxes),j),
				BOX_SCORE((output->boxes),j), BOX_SCALE((output->boxes),j),
				BOX_EXID((output->boxes),j),
				model->idMap[BOX_CLASS((output->boxes),j)].c_str());

		
			cv::rectangle(img,
					cv::Point(BOX_CMIN((output->boxes),j),BOX_RMIN((output->boxes),j)),
					cv::Point(BOX_CMAX((output->boxes),j),BOX_RMAX((output->boxes),j)),
					cv::Scalar(0,0,255),
					3,8,0
			);
		
	}
	
	cv::namedWindow("hog-pipeline",CV_WINDOW_AUTOSIZE);
	cv::imshow("hog-pipeline",img);
	cv::waitKey(0);
	cv::destroyWindow("hog-pipeline");
	cv::imwrite("detect-results.png",img);
	printf("Saved detections as detect-results.png\n");
	
	return 0;
}
