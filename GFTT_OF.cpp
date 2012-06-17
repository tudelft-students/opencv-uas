/* Good Featues To Track and Optical Flow
 * GFTT_OF.cpp
 *
 *  Created on: May 10, 2012
 *      Author: h2w
 */
//=== Compile in terminal with:
//g++ `pkg-config --cflags opencv` -o test.bin GFTT_OF.cpp `pkg-config --libs opencv`
//=== Plot the results in real-time plotter
//./test.bin|~/open_svn/driveGnuPlots.pl 2 50 50 X Y 600x400+0+0 -

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <stdio.h>

using namespace std;

#include <sys/time.h>
#define USEC_PER_SEC 1000000L

#define VIDEO_WINDOW   "Optic Flow"
#define FEAT_WINDOW   "Corners"

//=== returns the number of usecs of (t2 - t1)
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

int main(int argc, char *argv[]) {
	//=== loop counter
	//	int k = 0;

	//logfile=fopen("log_file","w");
	//long timestamp=0;

	//=== Initialize frame capture from camera
	CvCapture* capture = 0;
	IplImage* curr_frame = 0; // current video frame
	IplImage* curr_gray_frame = 0; // grayscale version of current frame
	IplImage* prev_frame = 0; // previous video frame
	IplImage* prev_gray_frame = 0; // grayscale version of previous frame
	IplImage* eig_image = 0;
	IplImage* temp_image = 0;
	capture = cvCaptureFromCAM(0);
	//cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,640);
	//cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,480);
	prev_frame = cvQueryFrame(capture);
	prev_gray_frame = cvCreateImage(cvGetSize(prev_frame),IPL_DEPTH_8U, 1);
	cvCvtColor(prev_frame, prev_gray_frame, CV_BGR2GRAY);

	//	start_timer();

	//=== cvGoodFeaturesToTrack Parameters
	const int MAX_CORNERS = 100; //5000
	double quality_level = 0.3; //0.2
	double min_distance = 1.0; //0.5
	int eig_block_size = 3; //8

	//=== cvFindCornerSubPix & cvCalcOpticalFlowPyrLK Parameters
	int win_size = 15;
	//	float xT = 0.0, yT = 0.0, x_ave = 0.0, y_ave = 0.0;

	//=== median filter Parameters
	float temp;
	int middle=0;
	float xmedian=0.0, ymedian=0.0;

	while (1) {

		//		timestamp=end_timer();
		//		fprintf(logfile,"%ld\n",timestamp);
		//		start_timer();

		//=== Get frame size
		CvSize img_sz = cvGetSize(prev_gray_frame);

		//=== Capture Subsequent image frames
		curr_frame = cvQueryFrame(capture);
		if( ! curr_gray_frame ) {
			curr_gray_frame = cvCreateImage(
					cvGetSize(curr_frame),
					IPL_DEPTH_8U, 1);
		}

		cvCvtColor(curr_frame, curr_gray_frame, CV_BGR2GRAY);

		// ==== Allocate memory for corner arrays ====
		if ( !eig_image) {
			eig_image = cvCreateImage(img_sz,
					IPL_DEPTH_32F, 1);
		}
		if ( !temp_image) {
			temp_image = cvCreateImage(img_sz,
					IPL_DEPTH_32F, 1);
		}

		//=== Good Features To Track Corner Detection
		int corner_count = MAX_CORNERS;
		CvPoint2D32f* cornersA = new CvPoint2D32f[ MAX_CORNERS ];
		cvGoodFeaturesToTrack(prev_gray_frame,
				eig_image,                    // output
				temp_image,
				cornersA,
				&corner_count,
				quality_level,
				min_distance,
				NULL,
				eig_block_size,
				false);

		cvFindCornerSubPix(
				prev_gray_frame,
				cornersA,
				corner_count,
				cvSize(win_size,win_size),
				cvSize(-1,-1),
				cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03)
				);

		//=== Draw the corners in color frame and Display in a window
		for( int i = 0; i < corner_count; i++) {
			int radius = prev_frame->height/50;
			cvCircle(prev_frame,
					cvPoint((int)(cornersA[i].x + 0.5f),(int)(cornersA[i].y + 0.5f)),
					radius,
					cvScalar(0,0,255,0));
		}
		cvNamedWindow(FEAT_WINDOW, 0); // allow the window to be resized
		cvShowImage(FEAT_WINDOW, prev_frame);

		//=== Call The Lucas Kanade Algorithm
		char features_found[ MAX_CORNERS ];
		float feature_errors[ MAX_CORNERS ];

		CvSize pyr_sz = cvSize( prev_gray_frame->width+8, curr_gray_frame->height/3 );

		IplImage* pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
		IplImage* pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );

		CvPoint2D32f* cornersB = new CvPoint2D32f[ MAX_CORNERS ];

		cvCalcOpticalFlowPyrLK(
				prev_gray_frame,
				curr_gray_frame,
				pyrA,
				pyrB,
				cornersA,
				cornersB,
				corner_count,
				cvSize( win_size,win_size ),
				2,
				features_found,
				feature_errors,
				cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ),
				0 );

		prev_gray_frame = cvCloneImage(curr_gray_frame);

		//=== Compute the Optical Flow & Draw them in color frame and Display in a window
		float dx[corner_count];
		float dy[corner_count];
		for( int i = 0; i < corner_count; i++) {
			dx[i]=cornersB[i].x-cornersA[i].x;
			dy[i]=cornersB[i].y-cornersA[i].y;
//			xT += dx[i];
//			yT += dy[i];
			CvPoint p0 = cvPoint( cvRound( cornersA[i].x ), cvRound( cornersA[i].y ) );
			CvPoint p1 = cvPoint( cvRound( cornersB[i].x ), cvRound( cornersB[i].y ) );
			cvLine( curr_frame, p0, p1, CV_RGB(255,0,0), 2 );
		}
		cvNamedWindow(VIDEO_WINDOW, 0); // allow the window to be resized
		cvShowImage(VIDEO_WINDOW, curr_frame);

		//=== Average Optical Flow
//		x_ave = xT/corner_count;
//		y_ave = yT/corner_count;

		//=== Median Optical Flow Approach: Sorting and Median Selection
		for(int n=0;n<(corner_count-1);n++){
			for(int m=n+1;m<corner_count;m++){
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
	    switch(corner_count%2) {
	        case(0): // even
	            middle = corner_count/2;
	        	xmedian = 0.5*(dx[middle-1]+dx[middle]);
	        	ymedian = 0.5*(dy[middle-1]+dy[middle]);
	        break;
	        case(1): // odd
	            middle = corner_count/2;
	        	xmedian = dx[middle];
	        	ymedian = dy[middle];
	        break;
	    }

	    //=== GnuPlots selected optical flow
		printf("0:%f\n",xmedian);
		printf("1:%f\n",ymedian);
		fflush(stdout);

		if ( (cvWaitKey(10) & 255) == 27)
			break;

//		k += 1;
//		if (k==100) break;

	} //end loop

	// Release the capture device housekeeping
	cvReleaseCapture( &capture);
	cvDestroyWindow(VIDEO_WINDOW);
	cvDestroyWindow(FEAT_WINDOW);
	//fclose(logfile);
	return 0;
}

