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

// These are the header file containing the prototype functions to be implemented by 
// Tools.cpp

// Tools.h



#include<string>
#include<fstream>

#ifndef TOOLS_LOCK
#define TOOLS_LOCK

namespace tools_namespace
{

	extern const std::string WHITESPACE;

	int		getInt		( std::string prompt );
	float	getFloat	( std::string prompt );
	char	getChar		( std::string prompt );
	std::string	getString	( std::string prompt );
	template <class Item> Item getStreamExtraction (std::string prompt, Item itemForTypeResolution, std::string itemName ); 

	char getBoundedChar( std::string prompt, char lowerBound, char upperBound );
	char getCharInString( std::string prompt, std::string chars );

	std::string	getLine				( std::string prompt );
	bool	getBool				( std::string prompt );
	int		getNonNegativeInt	( std::string prompt );
	int		getPositiveInt		( std::string prompt );
	int		getBoundedInt		( std::string prompt, int lowerBound, int upperBound );
	int		getPercent			( std::string prompt );
	int getBoundedIntWithErrorMessage( std::string prompt, int lowerBound, int upperBound, std::string errorMessage );

	void pause( std::string prompt = "Press ENTER to continue..." );
	void flushInput(void);

	char	lowerCase( char c );
	std::string	lowerCase( std::string s );

	bool isIn			( std::string pattern, std::string source );
	bool isInCaseless	( std::string pattern, std::string source );
	bool isWhitespace	( char c );
	bool isEmpty		( std::string s );

	void	chomp		( std::string &s );
	std::string	swab		( char c, int length );
	std::string	extractWord	( std::string &source );
	std::string	substitute	( std::string source, std::string pattern, std::string replacement );
	std::string	pluralize	( int count, std::string singular, std::string plural );

	void swap( int &a, int &b );
	int numberWidth( int n );
	int randomInt( int lowerBound, int upperBound );

	void openIfStream( std::ifstream &inStream, std::string prompt );
	void openOfStream( std::ofstream &outStream, std::string prompt );
}

#endif

