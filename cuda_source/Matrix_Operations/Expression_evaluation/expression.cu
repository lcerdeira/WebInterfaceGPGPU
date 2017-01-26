/* Author: H.M.Gamaarachchi (E/10/102)
 * C FILE POSTFIX SOLVING OF AN EXPRESSION 
 * list type used : arraylist 
 * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "stack.h"
#include "cudaarrayadd.cu"
#include "cudaarraysub.cu"
#include "multicuda.cu"
#include "inverseCUDA.cu"
#include "cpu_mat_op.cuh"
#include "infix2postfix.cu"

#define SEP " "
#define TMP_SIZE 10

int validchar(char ch){
	int flag;
	switch (ch){
		case 'A':
			flag=1;
			break;
		case 'B':
			flag=1;
			break;
		case 'C':
			flag=1;
			break;
		case 'D':
			flag=1;
			break;
		case 'E':
			flag=1;
			break;
		case ' ':
			flag=1;
			break;
		case '\t':
			flag=1;
			break;
		case '+':
			flag=1;
			break;
		case '-':
			flag=1;
			break;
		case '*':
			flag=1;
			break;
		case '(':
			flag=1;
			break;
		case ')':
			flag=1;
			break;
		case 'i':
			flag=1;
			break;
		case 'n':
			flag=1;
			break;
		case 'v':
			flag=1;
			break;
		default:
			flag=0;
			fprintf(stderr, "Not a valid expression\nCharacters allowed are A B C D E + - * ( ) inverses must be stated as inv(matrix)\n");
			exit(EXIT_FAILURE);
	}
	return flag;
}

//return the index when the matrix letter is given
int map_matrix(char id){
	int mapped;
	switch (id){
		case 'A':
			mapped=0;
			break;
		case 'B':
			mapped=1;
			break;
		case 'C':
			mapped=2;
			break;
		case 'D':
			mapped=3;
			break;
		case 'E':
			mapped=4;
			break;
		case 'F': //inv(A)
			mapped=5;
			break;
		case 'G':
			mapped=6;
			break;
		case 'H':
			mapped=7;
			break;
		case 'I':
			mapped=8;
			break;
		case 'J':
			mapped=9;
			break;
		
	}
	return mapped;
}
	
//remove whitespaces in the expression and check for invalid expressions
void removeSpaces(char *expr){
	int n=strlen(expr);
	char *temp=(char*)malloc(sizeof(char)*STR);
	int i=0,j=0;
	char current=expr[0];
	for(i=0;i<n && expr[i]!='\0';i++){
		current=expr[i];
		if (current!=' ' && current!='\t'){
			if(validchar(current)){
				temp[j]=current;
				j++;
			}
		}
	}
	temp[j]='\0';
	strcpy(expr,temp);
	//printf("%s\n", expr);
	return;
}

//find which inverses are necessary and replacing inv(A) by F inv(B) by G and so on 
void identify_inv(char *expr,int inv_flag[5]){
	int i=0,j=0;
	char *dest=(char*)malloc(sizeof(char)*STR);
	char ch=expr[i];
	char mat;
	while((ch=expr[i])!='\0'){
		if(ch!='i'){
			dest[j]=expr[i];
			i++;j++;
		}
		else{
			mat=expr[i+4];// matrix to be inverted
			if (expr[i+5] !=')'){
				fprintf(stderr,"Inverses of form inv(A) is supported.. Still expressions like inv(A+B) and inv(inv(A)*B) are not supported\n");
				exit(EXIT_FAILURE);
			}
			inv_flag[map_matrix(mat)]=1; //set that specific matrix needs to be inverted
			dest[j]=mat+5; //replace inv(A) by F, inv(B) by G and so on 
			j++;
			i=i+6; //skip inv(A) part
		}
	}
	dest[j]='\0';
	strcpy(expr,dest);
	//printf("%s\n", expr);
	return;
}



/* return 1 if an expression is erroneous*/
int error(STACK stack){
	if (stackcount(stack)<2)
		return 1;
	else 
		return 0;
}

	

int main(int argc, char *argv[]){

//Time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	
	
//checking args and getting args
	if(argc<18){
		fprintf(stderr,"Please enter all args eg: ./op A.txt rowA colA  B.txt rowB colB  C.txt rowC colC  D.txt rowD colD  E.txt rowE colE expression ans.txt");
		exit(1);
		}
		
	//stores input matrix width and height
	int width[TMP_SIZE];
	int height[TMP_SIZE];
	
	//names of input textfiles
	char matA[256],matB[256],matC[256],matD[256],matE[256];
	
	//get cmd args
	strcpy(matA,argv[1]);
	height[0]=atoi(argv[2]);
	width[0]=atoi(argv[3]);
	strcpy(matB,argv[4]);
	height[1]=atoi(argv[5]);
	width[1]=atoi(argv[6]);
	strcpy(matC,argv[7]);
	height[2]=atoi(argv[8]);
	width[2]=atoi(argv[9]);
	strcpy(matD,argv[10]);
	height[3]=atoi(argv[11]);
	width[3]=atoi(argv[12]);
	strcpy(matE,argv[13]);
	height[4]=atoi(argv[14]);
	width[4]=atoi(argv[15]);

	//read matrizes
	MATRIX matrix[TMP_SIZE];
	if (strcmp(matA,"NULL")!=0)
		matrix[0]=createMat(width[0],height[0],matA);
	if (strcmp(matB,"NULL")!=0)
	matrix[1]=createMat(width[1],height[1],matB);
	if (strcmp(matC,"NULL")!=0)
	matrix[2]=createMat(width[2],height[2],matC);
	if (strcmp(matD,"NULL")!=0)
	matrix[3]=createMat(width[3],height[3],matD);
	if (strcmp(matE,"NULL")!=0)
	matrix[4]=createMat(width[4],height[4],matE);
	
	MATRIX tmpmat; //temporary matrix for intermediate calculations
	//int tmp_count=10; //start of temp matrices
	
	//stack for solution
	STACK stack=createstack();
		
	char infix[256];
	strcpy(infix,argv[16]);
	removeSpaces(infix);
	
	int inv_flag[5]={0}; //flags to keep track of which matrices must be inverted
	identify_inv(infix,inv_flag); //identify matrices necessary to be inverted and replace them by chars
	
	//evaluate necessary inverses
	int  i;
	for (i=0;i<5;i++){
		if(inv_flag[i]==1){
			tmpmat=(MATRIX)malloc(sizeof(struct matrix));
			tmpmat->height=matrix[i]->height; 
			tmpmat->width=matrix[i]->width;
			if(tmpmat->height!=tmpmat->width){
				fprintf(stderr,"Inverses only exist for square matrices\n");
				exit(EXIT_FAILURE);
			}
			tmpmat->flag=1; //to identify as a undeletable matrix
			tmpmat->mat=(float*)malloc(sizeof(float)*tmpmat->height*tmpmat->width);
			inverse(tmpmat->mat,matrix[i]->mat,matrix[i]->height);
			checkCudaError(cudaGetLastError());
			matrix[i+5]=tmpmat;
		}
	}
	
	//convert to postfix
	char expr[256];
	infix_to_postfix(infix,expr);

	//char *word;
	//for (word = strtok(expr, SEP); word; word = strtok(NULL, SEP)){
	char val; i=0;
	for (i=0;i<STR && expr[i]!='\0';i++){
		val = expr[i];
		MATRIX operand2,operand1;
		
		//switch (word[0]){
		switch (val){
			case '+':
				if (error(stack)){
					puts("Invalid Expression!");
					exit(EXIT_FAILURE);
					}
				operand2=pop(stack);
				operand1=pop(stack);
				if(operand1->height!=operand2->height || operand1->width!=operand2->width){
					fprintf(stderr,"Error in addition. Dimension mismatch.\n");
					exit(EXIT_FAILURE);
				}
				tmpmat=(MATRIX)malloc(sizeof(struct matrix));
				tmpmat->height=operand1->height; 
				tmpmat->width=operand1->width;
				tmpmat->flag=0; //to identify it a deletable matrix
				tmpmat->mat=(float*)malloc(sizeof(float)*operand1->height*operand1->width);
				checkCudaError(add(tmpmat->mat,operand1->mat,operand2->mat,operand1->width,operand1->height));
				push(stack,tmpmat);
				freeMat(operand2);
				freeMat(operand1);
				//tmp_count++;
				break;
				
			case  '-':
				if (error(stack)){
					puts("Invalid Expression!");
					exit(1);
					}
				operand2=pop(stack);
				operand1=pop(stack);
				if(operand1->height!=operand2->height || operand1->width!=operand2->width){
					fprintf(stderr,"Error in subtraction. Dimension mismatch.\n");
					exit(EXIT_FAILURE);
				}
				tmpmat=(MATRIX)malloc(sizeof(struct matrix));
				tmpmat->height=operand1->height; 
				tmpmat->width=operand1->width;
				tmpmat->flag=0;
				tmpmat->mat=(float*)malloc(sizeof(float)*operand1->height*operand1->width);
				checkCudaError(substract(tmpmat->mat,operand1->mat,operand2->mat,operand1->width,operand1->height));
				push(stack,tmpmat);
				freeMat(operand2);
				freeMat(operand1);
				//tmp_count++;
				break;
			
			case  '*':
				if (error(stack)){
					fprintf(stderr,"Invalid Expression!");
					exit(1);
					}
				operand2=pop(stack);
				operand1=pop(stack);
				if(operand1->width!=operand2->height){
					fprintf(stderr,"Error in multiplication. Dimension mismatch.\n");
					exit(EXIT_FAILURE);
				}
				tmpmat=(MATRIX)malloc(sizeof(struct matrix));
				tmpmat->height=operand1->height; 
				tmpmat->width=operand2->width;
				tmpmat->flag=0;
				tmpmat->mat=(float*)malloc(sizeof(float)*tmpmat->height*tmpmat->width);
				multi(tmpmat->mat,operand1->mat,operand1->width,operand1->height,operand2->mat,operand2->width,operand2->height);
				checkCudaError(cudaGetLastError());
				push(stack,tmpmat);
				freeMat(operand2);
				freeMat(operand1);
				//tmp_count++;
				break;
			
			/*case  '/':
				if (error(stack)){
					puts("Invalid Expression!");
					exit(1);
					}
				operand2=pop(stack);
				operand1=pop(stack);
				temp=operand1/operand2;
				push(stack,temp); 
				break;*/
				
				
			default:
				//val = word[0];
				push(stack,matrix[map_matrix(val)]);  
				break;
		}
	}
	
	if (stackcount(stack)==1){
		MATRIX result=pop(stack); // the result
		saveMat(result, argv[17]);
	}
	else{
		fprintf(stderr,"Invalid Expression!");
		exit(1);
	}
	destroystack(stack);
	
	//Time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for calculation : %.10f\n",elapsedtime/(float)1000);
	return 0;
}
