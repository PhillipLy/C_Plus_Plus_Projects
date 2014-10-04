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

// This is the header file of the TaskDriver. It provides the necessary prototype functions 
// so that they can be implemented by the TaskDriver.cpp.

// TaskDriver.h



#include"Stack.h"
#include"Task.h"

namespace task_driver_namespace
{
	typedef stack_namespace::Stack<task_namespace::Task> TaskStack;

	void taskDriver(void);

	// called by taskDriver:
	void addTask		( TaskStack &tasks, time_t current_time );
	void advanceTime	( TaskStack &tasks, time_t &current_time );
}