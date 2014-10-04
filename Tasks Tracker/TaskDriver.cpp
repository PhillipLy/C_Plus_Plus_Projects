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


// This is the main implementation file for the Tasks Tracker program; it includes the 
// implementation for most of what the program should do as outlined in the overview of
// the program's functions.

// TaskDriver.cpp



#include"TaskDriver.h"
#include"Task.h"
#include"Stack.h"
#include"Tools.h"
#include<iostream>
using namespace tools_namespace;
using namespace stack_namespace;
using namespace task_namespace;
using namespace std;

namespace task_driver_namespace
{
	void taskDriver(void)
	{
		TaskStack tasks;

		// walk through time
		for (time_t currentTime = 0 ; true ; advanceTime(tasks,currentTime) )
		{
			// state
			cout << endl;
			cout << "Time: " << currentTime << endl;
			if ( tasks.isEmpty() )
				cout << "No active task." << endl;
			else
				cout << "Active Task: " << tasks.top().description() << endl;

			// action
			if ( getBool("Add a task? ") )	addTask(tasks,currentTime);
			else if ( getBool("Quit? ") )	break;
		}
	}

	void addTask( TaskStack &tasks, time_t currentTime )
	{
		// suspend currently-executing task
		if ( !tasks.isEmpty() )
		{
			cout << "Interrupting task: " << tasks.top().description() << endl;
			tasks.top().deactivate(currentTime);
		}

		// install new task
		string description	= getLine			( "Enter description of new task: " );
		size_t duration		= getPositiveInt	( "Enter duration of new task:     " );
		tasks.push( Task(currentTime,description,duration) );
		tasks.top().activate(currentTime);
	}

	void advanceTime( TaskStack &tasks, time_t &currentTime )
	{
		cout << endl;

		// walk through some time units
		for ( size_t toElapse = getPositiveInt("Enter the time elapsed for the current task: ") ; toElapse > 0 ; --toElapse )
		{
			++currentTime;

			// see if current task has completed
			if ( !tasks.isEmpty() && tasks.top().isFinished(currentTime) )
			{
				// finish up current task
				cout << "Total time elapsed for all tasks so far: " << currentTime << endl;
				cout << "Task completed: " << tasks.pop().description() << endl;

				// restart task on top of stack
				if ( !tasks.isEmpty() )
				{
					cout << "Resuming task: " << tasks.top().description() << endl;
					tasks.top().activate(currentTime);
				}
			}
		}
	}
}