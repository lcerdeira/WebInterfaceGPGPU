/* Author: H.M.Gamaarachchi (E/10/102)
 * Lab 2
 * 30.07.2013 
 * C FILE FOR STACK DATA STRUCTURE */
 

#include "stack.h"

/*create a new stack*/
STACK createstack(void){
	STACK newstack=newlist();
	return newstack;
}

/*push an element to the stack*/
int push(STACK stack, ELEMENT element){
	int count=stack->count;
	insert(stack,count,element);
	return 1;
}

/*pop out the last element from the stack*/
ELEMENT pop(STACK stack){
	int count=stack->count-1;
	ELEMENT element=retrieve(stack,count);
	deleteitem(stack,count);
	return element;
}

/*peek into the last element in the stack*/	
ELEMENT top(STACK stack){
	int count=stack->count-1;
	ELEMENT element=retrieve(stack,count);
	return element;
}

/*return the count of a stack*/
int stackcount(STACK stack){
	return stack->count;
}

/*destroy a stack*/
void destroystack(STACK stack){
	destroylist(stack);
}


