/* H.M. Gammarachchi
 * E/10/102
 * LAB 01
 * */

#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "arraylist.h"

/**
	Return a dummy header
**/
LIST newlist(void) 
{
	LIST array = (LIST)malloc(sizeof(struct ARRAY));
	array-> data = (ELEMENT*)malloc(sizeof(ELEMENT) * MAX);
	array->count = 0;
	return array;
}

/**
	Free the memory used by the list
	Note the order of operations
**/
void destroylist(LIST list) 
{
	free(list->data);
	free(list);	
}

ELEMENT retrieve (LIST list,int index)
{
	if (index<(list->count) && index>=0){ /*condition for invalid index*/
		return (list->data[index]);
		}
	return NULL; //error
}

/**
	insert an element 
**/
LIST insert(LIST list, int index, ELEMENT element)
{	
	int i=list->count-1;
	if (index>=0 && index<=list->count && list->count<MAX){
		while(i>=index){ /*moving elements to get space*/
			list->data[i+1]=list->data[i];
			i--;
		}
		list->data[index]=element; /*storing*/
		list->count++;
	}
	return list;
}

LIST deleteitem (LIST list,int index) 
{
	int i=index;
	if (index>=0 && index<list->count){
		while (i<list->count-1){ /*moving elements and deleting*/
			list->data[i]=list->data[i+1];
			i++;
		}
		list->count--;
	}
	return list;
}

int search(LIST list, ELEMENT element)
{
	int i=0;
	for(i=0;i<list->count;i++){
		if (list->data[i]==element){
			return i;
		}
	}
	return -1;
}


