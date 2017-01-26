/*
Author : H.M.Gamaarachchi
Mandelbrot set in CUDA
command line arguments are  WIDTH HEIGHT REAL_MIN REAL_MAX IMAGINARY_MIN IMAGINARY_MAX OUT.txt OUT.ppm
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "myhelpers.h"

#define BLOCK 16
#define INF 4
#define MAXN 3000
#define max(a,b) (((a)>(b))?(a):(b))
	



//Tranform a pixel to complex plane
__device__ void transform_to_x(int x,float *x_dev,int WIDTH,float XMIN,float XMAX){ 
	*x_dev=(XMIN+x*(XMAX-XMIN)/(float)WIDTH);
}
__device__ void transform_to_y(int y,float *y_dev,int HEIGHT,float YMIN,float YMAX){
	*y_dev=(YMAX-y*(YMAX-YMIN)/(float)HEIGHT);
}


//check whether is in mandelbrot set
__device__ void isin_mandelbrot(float realc,float imagc,int *ans){
	int i=0;
	float realz_next=0,imagz_next=0;
	float abs=0;
	float realz=0;
	float imagz=0;
	while(i<MAXN && abs<INF){
		realz_next=realz*realz-imagz*imagz+realc;
		imagz_next=2*realz*imagz+imagc;
		abs=realz*realz+imagz*imagz;
		realz=realz_next;
		imagz=imagz_next;
		i++;
	}
	if (i==MAXN)
		*ans= 0;
	else
		*ans= i;
}

unsigned char red(int i){
	if (i==0 )
		return 0 ;
	else 
	return ((i+10)%256);
}

/*Calculate B value in RGB based on divergence*/
unsigned char blue(int i){
	if (i==0)
	return  0;
	else
	return ((i + 234) % 7 * (255/7));
}

/*Calculate G value in RGB based on divergence*/
unsigned char green(int i){
	if (i==0)
		return  0 ;
	else
		return ((i+100) % 9 * (255/9));
}

//Make the plotting matrix of colors depending on the presence in mandelbrot
__global__ void plot(int *blank,int WIDTH,int HEIGHT,float XMIN,float XMAX,float YMIN,float YMAX){
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	if (x<WIDTH && y<HEIGHT){
		int n=y*WIDTH+x;
		float x_trans;
		float y_trans;
		int ans;
		transform_to_x(x,&x_trans,WIDTH,XMIN,XMAX);
		transform_to_y(y,&y_trans,HEIGHT,YMIN,YMAX);
		isin_mandelbrot(x_trans,y_trans,&ans);
		blank[n]=ans;		
	}
}

//create the imahe matrix
void createimage(int *mandel_set,unsigned char *image,int WIDTH,int HEIGHT) {

  	int x=0,y=0,n=0;int color;
	for (y=0;y<HEIGHT;y++){
		for(x=0;x<WIDTH;x++){
			color=mandel_set[y*WIDTH+x];
			image[n]=red(color);
			image[n+1]=green(color);
			image[n+2]=blue(color);
			n=n+3;
		}
	}	

}


int main(int argc, char** argv) {

  //check values
  if (argc<9){
	fprintf(stderr,"Enter arguments as ./binary WIDTH HEIGHT REAL_MIN REAL_MAX IMAGINARY_MIN IMAGINARY_MAX OUT.txt out.ppm\n");
	exit(EXIT_FAILURE);
  }
  
  int WIDTH=atoi(argv[1]);
  int HEIGHT=atoi(argv[2]);
  float XMIN=atof(argv[3]);
  float XMAX=atof(argv[4]);
  float YMIN=atof(argv[5]);
  float YMAX=atof(argv[6]);	

  printf("\nwidth : %d  height : %d  xmin : %f  xmax:%f  ymin : %f  ymax : %f \n",WIDTH,HEIGHT,XMIN,XMAX,YMIN,YMAX);

  //Memory allocation
  int *dev_mandel;
  checkCudaError(cudaMalloc((void**)&dev_mandel, HEIGHT* WIDTH * sizeof(int)));

  //CUDA function calling
  float tdx=(float)BLOCK; //max possible threads per block
  float tdy=(float)BLOCK;
  dim3 grid(ceil(WIDTH/tdx),ceil(HEIGHT/tdy));
  dim3 block(tdx,tdy);
 

//time calculations

  cudaEvent_t start,stop;
  float elapsedtime;
  cudaEventCreate(&start);
  cudaEventRecord(start,0);
  
  plot<<<grid, block>>>(dev_mandel,WIDTH,HEIGHT,XMIN,XMAX,YMIN,YMAX);
  checkCudaError(cudaGetLastError());

//time calculation
  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedtime,start,stop);
  printf("Time spent for calculation in CUDA : %.10f s\n",elapsedtime/(float)1000);

  //copy back and clear 
  int *mandel_set=(int *)malloc(sizeof(int)*WIDTH*HEIGHT);
  checkCudaError(cudaMemcpy(mandel_set, dev_mandel,  HEIGHT* WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(dev_mandel);

  FILE *fp;
  fp=fopen(argv[7],"w");
  isFileOK(fp);
  //printing results
  int x,y;
  for (y=0;y<HEIGHT;y++){
	  for(x=0;x<WIDTH;x++){
		  fprintf(fp,"%d ",mandel_set[y*WIDTH+x]);
	//printf("%d ",mandel_set[y][x]);
	  }
	  fprintf(fp,"\n");
 //printf("\n");
	  }
  fclose(fp);

 //Getting image
  unsigned char *image=(unsigned char *)malloc(sizeof(unsigned char)*WIDTH*HEIGHT*3);
  createimage(mandel_set,image,WIDTH,HEIGHT);

 //Writing jpg
   // color component ( R or G or B) is coded from 0 to 255 
        // it is 24 bit color RGB file 
        const int MaxColorComponentValue=255; 
        char *filename=argv[8];
        char *comment="# ";//comment should start with # 
        //unsigned char color[3];
        
        //create new file,give it a name and open it in binary mode  
        fp= fopen(filename,"wb"); // b -  binary mode 
		isFileOK(fp);
        //write ASCII header to the file
        fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment,WIDTH,HEIGHT,MaxColorComponentValue);
        // compute and write image data bytes to the file
        fwrite(image,1,WIDTH *HEIGHT * 3,fp);
			
        fclose(fp);

  free(mandel_set);
  free(image);
  return 0;
}
