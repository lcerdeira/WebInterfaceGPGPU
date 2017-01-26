//free a matrix if it is not an input
void freeMat(MATRIX matrix){
	if(matrix->flag==0){
		free(matrix->mat);
		free(matrix);
	}
}

//create a matrix using an input textfile
MATRIX createMat(int width, int height, char *filename){
	//allocating
	MATRIX matrix=(MATRIX)malloc(sizeof(struct matrix));
	isMemoryFullstruct(matrix);
	matrix->height=height;
	matrix->width=width;
	matrix->flag=1;
	matrix->mat=(float *)malloc(width*height*sizeof(float));
	isMemoryFull(matrix->mat);
	
	//reading files
	int i;
	FILE *fp;
	fp=fopen(filename,"r");
	isFileOK(fp);
	for (i=0;i<width*height;i++){
		fscanf(fp,"%f",&(matrix->mat[i]));
		}
	fclose(fp);
	return  matrix;
}

//save a matrix into a text file
void saveMat(MATRIX matrix,char *filename){
	int i,j;
	int width=matrix->width;
	int height=matrix->height;
	FILE *fp=fopen(filename,"w");
	isFileOK(fp);
	for (i=0;i<height;i++){
		for (j=0;j<width;j++){
			fprintf(fp,"%f ",matrix->mat[width*i+j]);
		}	
		fprintf(fp,"\n");
	}
	fclose(fp);
}