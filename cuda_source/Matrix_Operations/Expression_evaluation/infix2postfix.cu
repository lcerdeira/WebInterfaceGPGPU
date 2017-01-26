/* infix to postfix expression conversion code
by Codingstreet.com */
#include<stdio.h>
#include<string.h>

int push(char *stack,char val,int *top,int *size);
int pop(char *stack,int *top);
int isstack_empty(int *top);
int isstack_full(int *top,int *size);

int isstack_empty(int *top){
    if((*top)==0) return 1;
    return 0;
}
int isstack_full(int *top,int *size){
    if((*top)==(*size)-1) return 1;
    return 0;
}
int push(char *stack,char val,int *top,int *size){
    if(isstack_full(top,size)){
        return 0;
    }
    stack[(*top)++]=val;
    return 1;
}
int pop(char *stack,int *top){
    if(isstack_empty(top)){
        return -1;
    }
    else return stack[--(*top)];
}
int get_precedence(char c){
	switch(c){
		case '+':
		case '-':
		return 1;
		case '*':
		case '/':
		case '%':
		return 2;
		case '^':
		return 0;
		case '(':
		return -1;
		default :
		return -2;
}
}
void infix_to_postfix(char *instring,char *outstring){
	int i=0,top,size,pred1,pred2,n=0;
	char c,c2;
	int len=strlen(instring);
	if(instring==NULL) return;
	char *stack=(char*)malloc(sizeof(char)*(len-1));
	top=0;size=len-1;
	while(instring[i]!='\0'){
		c=instring[i];
			if(c==' ') {i++;continue; }
			else if(c=='('){
				push(stack,c,&top,&size);
			}
		else if(c=='+' || c=='-' || c=='*' || c=='/' || c=='%'||c=='^'){
			if(isstack_empty(&top)) {
				push(stack,c,&top,&size);
			}
			else {
				pred1=get_precedence(stack[top-1]);
				pred2=get_precedence(c);
				while(pred2<=pred1 && !isstack_empty(&top)){
					c2=pop(stack,&top);
					outstring[n]=c2;
					n++;
					pred2=get_precedence(stack[top-1]);
					pred1=get_precedence(c);
				}
			push(stack,c,&top,&size);
			}
		}
		else if(c==')'){
			while(stack[top-1]!='('){
				c2=pop(stack,&top);
				outstring[n]=c2;
				n++;
			}
			pop(stack,&top);
		}
		else {
			outstring[n]=c;
			n++;
		}
	i++;
	}
	
	while(!isstack_empty(&top)){
		c=pop(stack,&top);
		outstring[n]=c;
		n++;
	}
	
	outstring[n]='\0';
	
}

/*int main(){
	char str[]="((a+t)*((b+(a+c))^(c+d)))";
	char out[100];
	infix_to_postfix(str,out);
	printf("\nInput string :%s \nOutput: %s ",str,out);
	

	return 0;
}*/