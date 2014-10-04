// Tasks Tracker by Phillip Ly

/////////////////////Overview of the program's functions//////////////////////////////////

// This is a task tracking program that implements a template stack to keep track inputted 
// tasks. When a task is initiated (introduced to the stack through the user's inputs), it 
// is assigned with a name and an amount of time necessary to reach the status of completion.

// If there was a current active task on the stack, it is deactivated from the active slot
// (the active slot of task is the top position of the stack) of the stack before the new 
// task is introduced. This new task is then positioned on top of the stack while the
// old task is now at the position after the current task (second position from the top
// of the stack).

// Whenever a task is finished, it will be reported/outputted and deleted from the stack.
// If a task is removed from the stack, the most recently deactivated task (the task at the 
// second position from the top of the stack) should be reactivated and assume the top 
// position of the stack.

// This is the implementation file for main which control when the main program initiates or ends.
// main.cpp



#include"TaskDriver.h"

int main(void)
{
	task_driver_namespace::taskDriver();

	return 0;
}