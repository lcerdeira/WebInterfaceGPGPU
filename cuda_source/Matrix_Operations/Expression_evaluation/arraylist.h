#define MAX 1000
#define STR 256

struct matrix{
	int height;
	int width;
	float *mat;
	int flag; //1 if an input whicg cannot be freedup
};

typedef struct matrix *MATRIX;
typedef struct matrix *ELEMENT;

struct ARRAY
{
	int count;
	ELEMENT *data;
};

typedef struct ARRAY *LIST;

#include "list.h"
