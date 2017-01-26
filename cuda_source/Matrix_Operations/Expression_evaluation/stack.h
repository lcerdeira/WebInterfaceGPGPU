/* Author: H.M.Gamaarachchi (E/10/102)
 * Lab 2
 * 30.07.2013 
 * Header FILE FOR STACK DATA STRUCTURE */

#include "arraylist.h"
#include "list.h" 

typedef LIST STACK;

STACK createstack(void);
int push(STACK stack,ELEMENT element);
ELEMENT pop(STACK stack);
ELEMENT top(STACK stack);
int stackcount(STACK stack);
void destroystack(STACK stack);
