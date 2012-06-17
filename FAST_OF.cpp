/* FAST and Optical Flow
 * FAST_OF.cpp
 *
 *  Created on: May 31, 2012
 *      Author: h2w
 */
//=== Compile in terminal with:
//g++ `pkg-config --cflags opencv` -o test.bin FAST_OF.cpp `pkg-config --libs opencv`
//=== Plot the results in real-time plotter
//./test.bin|~/open_svn/driveGnuPlots.pl 2 50 50 X Y 600x400+0+0 -

#include <cvaux.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
//#include <opencv.hpp>
//#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/video/tracking.hpp>
#include <stdio.h>

using namespace std;

//=== g++ `pkg-config --cflags opencv` -o test.bin main.cpp `pkg-config --libs opencv`
//=== ./test.bin|~/open_svn/driveGnuPlots.pl 2 50 50 X Y 600x400+0+0 -

//=== returns the number of usecs of (t2 - t1)
#include <sys/time.h>
#define USEC_PER_SEC 1000000L
long time_elapsed (struct timeval &t1, struct timeval &t2) {
	long sec, usec;
	sec = t2.tv_sec - t1.tv_sec;
	usec = t2.tv_usec - t1.tv_usec;
	if (usec < 0) {
		--sec;
		usec = usec + USEC_PER_SEC;
	}
	return sec*USEC_PER_SEC + usec;
}
struct timeval start_time;
struct timeval end_time;
void start_timer() {
	struct timezone tz;
	gettimeofday (&start_time, &tz);
}
long end_timer() {
	struct timezone tz;
	gettimeofday (&end_time, &tz);
	return time_elapsed(start_time, end_time);
}
//FILE* logfile;

//=== check max & min bound
//float dx_max = 0.0, dx_min = 0.0, dy_max = 0.0, dy_min = 0.0;

int main(int argc, char *argv[]){
	//=== loop counter
	//	int k = 0;

	//=== log file
//	logfile=fopen("log_file","w");
//	long timestamp = 0;

	//=== Initialize frame capture from camera
	CvCapture* capture = 0;
	IplImage* curr_frame = 0; // current video frame
	IplImage* curr_gray_frame = 0;
	IplImage* prev_frame = 0; // previous video frame
	IplImage* prev_gray_frame = 0;
	capture = cvCaptureFromCAM(0);
//	cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,160);
//	cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,120);
	prev_frame = cvQueryFrame(capture);
	prev_gray_frame = cvCreateImage( cvGetSize( prev_frame ), IPL_DEPTH_8U, 1); // grayscale version of current frame
	cvCvtColor(prev_frame, prev_gray_frame, CV_RGB2GRAY);

//	start_timer();

	//=== FAST Parameters
	int inFASTThreshhold = 80;
	bool inNonMaxSuppression  = true;
	vector<cv::KeyPoint> corners;

	//=== cvCalcOpticalFlowPyrLK Parameters
	int win_size = 5;//15
//	float xT = 0.0, yT = 0.0, x_ave = 0.0, y_ave = 0.0;

	//=== median filter Parameters
	float temp;
	int middle=0;
	float xmedian=0.0, ymedian=0.0;

	while(1){

//		timestamp=end_timer();
//		fprintf(logfile,"%ld\n",timestamp);
//		start_timer();

		//=== Capture Subsequent image frames
		curr_frame = cvQueryFrame(capture);
		if (! curr_gray_frame ) {
			curr_gray_frame = cvCreateImage(
								cvGetSize(curr_frame),
								IPL_DEPTH_8U, 1);
		}

		cvCvtColor(curr_frame, curr_gray_frame, CV_RGB2GRAY);

		//corners.clear();

		//=== FAST corner detection
		FAST(cv::Mat((const IplImage*)prev_gray_frame,1), corners, inFASTThreshhold, inNonMaxSuppression);

		int numCorners = corners.size();
		if (numCorners == 0){
			xmedian = 0;
			ymedian = 0;
		}
		else{
			CvPoint2D32f* cornersA = new CvPoint2D32f[ numCorners ];
			//printf("%d\n",(int)numCorners);

			//=== Extract the corners & Draw them in color frame and Display in a window
			for (int i = 0; i < numCorners; ++i) {
				const cv::KeyPoint &kp = corners[i];
				cv::Point2f coordinates = kp.pt;
				cornersA[i].x = coordinates.x;
				cornersA[i].y = coordinates.y;
				int radius = curr_frame->height/50;
				cvCircle(curr_frame,
						cvPoint((int)(coordinates.x + 0.5f),(int)(coordinates.y + 0.5f)),
						radius,
						cvScalar(0,0,255,0));
			}
			cvNamedWindow("corners", 0); // allow the window to be resized
			cvShowImage("corners", curr_frame);

			//=== Call The Lucas Kanade Algorithm
			char features_found[ numCorners ];
			float feature_errors[ numCorners ];
			CvSize pyr_sz = cvSize( prev_gray_frame->width+8, curr_gray_frame->height/3 );
			IplImage* pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
			IplImage* pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
			CvPoint2D32f* cornersB = new CvPoint2D32f[ numCorners ];
			cvCalcOpticalFlowPyrLK(
					prev_gray_frame,
					curr_gray_frame,
					pyrA,
					pyrB,
					cornersA,
					cornersB,
					numCorners,
					cvSize( win_size,win_size ),
					2,
					features_found,
					feature_errors,
					cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ),
					0);

			//=== Compute the Optical Flow & Draw them in color frame and Display in a window
			float dx[numCorners];
			float dy[numCorners];
			for( int i = 0; i < numCorners; ++i) {
				dx[i] = cornersA[i].x - cornersB[i].x;
				dy[i] = cornersA[i].y - cornersB[i].y;
//				xT = xT + dx[i];
//				yT = yT + dy[i];
				CvPoint p0 = cvPoint( cvRound( cornersA[i].x ), cvRound( cornersA[i].y ) );
				CvPoint p1 = cvPoint( cvRound( cornersB[i].x ), cvRound( cornersB[i].y ) );
				cvLine( curr_frame, p0, p1, CV_RGB(255,0,0), 2 );
			}
			cvNamedWindow("opticflow", 0); // allow the window to be resized
			cvShowImage("opticflow", curr_frame);

			//=== Average Optical Flow
	//		x_ave = xT/numCorners;
	//		y_ave = yT/numCorners;

			//==== Median Optical Flow Approach: Sorting and Median Selection

			for(int n=0;n<(numCorners-1);n++){
				for(int m=n+1;m<numCorners;m++){
					if(dx[n]>dx[m]){
						temp=dx[n];
						dx[n]=dx[m];
						dx[m]=temp;
					}
					if(dy[n]>dy[m]){
						temp=dy[n];
						dy[n]=dy[m];
						dy[m]=temp;
					}
				}
			}
//			switch(numCorners%2) {
//				case(0): // even
//					middle = numCorners/2;
//					xmedian = 0.5*(dx[middle-1]+dx[middle]);
//					ymedian = 0.5*(dy[middle-1]+dy[middle]);
//				break;
//				case(1): // odd
						middle = numCorners/2;
						xmedian = dx[middle];
						ymedian = dy[middle];
//			    break;
//			}
//
//			    if (xmedian>dx_max)
//			    	dx_max = xmedian;
//			    if (xmedian<dx_min)
//			    	dx_min = xmedian;
//			    if (ymedian>dy_max)
//			    	dy_max = ymedian;
//			    if (ymedian<dy_min)
//			    	dy_min = ymedian;

		} //end else

		//=== GnuPlots selected optical flow
		printf("0:%f\n",xmedian);
		printf("1:%f\n",ymedian);

		prev_gray_frame = cvCloneImage(curr_gray_frame);
		fflush(stdout);

		if ( (cvWaitKey(10) & 255) == 27)
			break;
//		k++;
//		if (k>1000) break;

	} //end loop
	//printf("dx_max = %f & dx_min = %f\n",dx_max,dx_min);
	//printf("dy_max = %f & dy_min = %f\n",dy_max,dy_min);
	cvReleaseCapture( &capture);
	cvDestroyWindow("corners");
	cvDestroyWindow("opticflow");
//	fclose(logfile);
	return 0;
}




