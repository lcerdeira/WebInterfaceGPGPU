
/* CUDA C file for merge sort
Author : H.M.Gamaarachchi
*/

#include "mergesort_cuda.cuh"

int main(int argc, char *argv[]){
	//error checking on arguments
	if (argc<5){
		fprintf(stderr,"Enter all args eg : ./sort input.txt size 1 out.txt \n 1 stands for ascending 2 stands for descending");
		exit(EXIT_FAILURE);
	}

	//reading parameters and data
	int SIZE=atoi(argv[2]);
	int order=atoi(argv[3]);
	
	float *data=(float *)malloc(sizeof(float)*SIZE);
	isMemoryFull(data);
	float *sorted=(float *)malloc(sizeof(float)*SIZE);
	isMemoryFull(sorted);

	FILE *fp=fopen(argv[1],"r");
	isFileOK(fp);
	int i;
	for (i=0;i<SIZE;i++){
		fscanf(fp,"%f",&data[i]);
	}

	clock_t start=clock();
	//function calling
	sort(sorted,data,SIZE,order);
	clock_t stop=clock();
    double cputime=(double)((stop-start)/(float)CLOCKS_PER_SEC);
    printf("Time for total operation is %1.20f s\n",cputime);

	//writing data
	fp=fopen(argv[4],"w");
	isFileOK(fp);
	for (i=0;i<SIZE;i++){
		fprintf(fp,"%f\n",sorted[i]);
	}

	free(data);
	free(sorted);

	return 0;
}