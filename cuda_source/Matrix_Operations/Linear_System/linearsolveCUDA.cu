/*author : H.M.Gamaarachchi
c file for solving linear equations using LU decomposition
reference : http://rosettacode.org/wiki/LU_decomposition#C
http://www.ece.mcmaster.ca/~kiruba/3sk3/lecture6.pdf
 A.x=b solved by forward substitution L.d=b and the backward substitution U.x=d*/

#include "ludecompose_cuda.cuh"

//solve linear set of equations using LU decomposition, b is the coefficient vector
void linearSolve(float *ans, float *matL, float *matU,float *matb,float *tempd, int size){
	int i,j;
	float d,x;
	
	//forward substitution
	for (i=0;i<size;i++){
		d=matb[i]; //initially di=bi
		for (j=0;j<i;j++){
			d=d-matL[i*size+j]*tempd[j]; //di=bi-sigma(Lij*dj)
		}
		tempd[i]=d;
	}
	
	//backward substitution
	for (i=size-1;i>=0;i--){
		x=tempd[i]; //initially xi=di
		for (j=i+1;j<size;j++){
			x=x-matU[i*size+j]*ans[j]; //xi=di-sigma(Uij*xj)
		}
		ans[i]=x/matU[i*size+i]; //xi=xi/Uii
	}	
	
}

int main(int argc, char *argv[]){
	int size;
	
	//checking args and getting args
	if(argc<5){
		fprintf(stderr,"Please enter all args eg: ./Linear matA.txt matb.txt rows ans.txt \n where the format is Ax=b ");
		exit(EXIT_FAILURE);
		}
		
	size=atoi(argv[3]);
	
	//allocating
	float *matA=(float*)malloc(size*size*sizeof(float));
	isMemoryFull(matA);
	float *matL=(float*)malloc(size*size*sizeof(float));
	isMemoryFull(matL);
	float *matU=(float*)malloc(size*size*sizeof(float));
	isMemoryFull(matU);
	float *matb=(float*)malloc(size*sizeof(float));
	isMemoryFull(matb);
	float *tempd=(float*)malloc(size*sizeof(float));
	isMemoryFull(tempd);
	float *ans=(float*)malloc(size*sizeof(float));
	isMemoryFull(ans);
	
	//reading files
	int i;
	FILE *fp;
	fp=fopen(argv[1],"r");
	isFileOK(fp);
	for (i=0;i<size*size;i++){
		fscanf(fp,"%f",&matA[i]);
		}
	fclose(fp);
	//printf("reading mat A finished\n");
	
	fp=fopen(argv[2],"r");
	isFileOK(fp);
	for (i=0;i<size;i++){
		fscanf(fp,"%f",&matb[i]);
		}
	fclose(fp);
	//printf("reading mat b finished\n");

	// linear
	clock_t start=clock();
	
	LUdecompose(matL,matU,matA,size); //happen in CUDA
	checkCudaError(cudaGetLastError());
	
	linearSolve(ans, matL, matU,matb,tempd,size); //happen in cpu
	
	clock_t stop=clock();
    double cputime=(double)((stop-start)/(float)CLOCKS_PER_SEC);
    printf("Time for calculation with memory transfer overhead : %1.10f s\n",cputime);
	
	//writing to file
	fp=fopen(argv[4],"w");
	isFileOK(fp);
	for (i=0;i<size;i++){
		fprintf(fp,"%f ",ans[i]);
		fprintf(fp,"\n");
	}	
	fclose(fp);	
	
	
	return 0;

}
