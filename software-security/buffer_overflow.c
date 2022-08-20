#include <stdio.h>
#include <string.h>

void func(char *arg) {
  int authenticated;
  authenticated =0;
  char buffer[4];
  strcpy(buffer, arg);
  if(authenticated) {
    printf("Authenticated :)\n");
    return;
  }
  printf("Not authenticated :(\n");
  return;
}


int main() {
  char *myStr = "Auth";
  func(myStr);
  return 0;
}
