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

// This is the task class, its primary role is to keep track of the time it takes for a 
// task to be completed. It is designed with methods to show how much time has elapsed.


// Task.h


#include<cassert>
#include<string>

#ifndef TASK_LOCK
#define TASK_LOCK

namespace task_namespace
{
	class Task
	{
	public:
		Task( time_t creationTime, std::string description, time_t timeRemaining )
			: taskLastTimestamp		(creationTime),
				taskDescription		(description),
				taskTimeRemaining	(timeRemaining),
				taskIsActive		(false)
			{}

		std::string description(void) const {return taskDescription; }

		bool isActive			(time_t currentTime) { update(currentTime); return taskIsActive; }
		bool isFinished			(time_t currentTime) { update(currentTime); return taskTimeRemaining == 0; }
		time_t timeRemaining	(time_t currentTime) { update(currentTime); return taskTimeRemaining; }
		void deactivate			(time_t currentTime) { update(currentTime); taskIsActive = false; }
		void activate			(time_t currentTime) { if (!isFinished(currentTime)) taskIsActive = true; }

	private:
		time_t taskLastTimestamp;
		time_t taskTimeRemaining;
		std::string taskDescription;
		bool taskIsActive;

		void update( time_t currentTime )
		{
			assert( currentTime >= taskLastTimestamp );
			time_t elapsed = currentTime - taskLastTimestamp;
			taskLastTimestamp = currentTime;
			if ( !taskIsActive ) return;
			if ( elapsed < taskTimeRemaining )
				taskTimeRemaining -= elapsed;
			else
			{
				taskTimeRemaining = 0;
				taskIsActive = false;
			}
		}
	};
}

#endif