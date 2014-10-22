//Friends Ranking Manipulator
//Phillip Ly

//Capabilities of the program

/* 1) Add people to the list
   2) Delete people from the list
   3) Change rankings of people on the list (The program should make sure that 
      rankings are always unique and consecutive, counting up from 1.)
   4) Display the list, ordered by either name or rank
   5) Write the list to a file
   6) Read the list from a file
   7) Display a single friend, specified by either name or rank
   8) Display a statistic about what percent of names are long (A friend’s name 
      is long if it has more letters in it than his/her rank)
*/

// Valid commands to enter are: read, write, add, lookup, delete, rank, stat, print, quit


#include<iostream>
#include<fstream>
#include<string>
#include<vector>
using namespace std;

///////////////////////
// Database definition
///////////////////////

class NameRank
{
public:
	NameRank( string i_name, int i_rank ) : name(i_name), rank(i_rank) {} 
	string	name;
	int		rank;
};

class NameRankDatabase
{
public:
	bool	readRankedNames		( string	fileName );
	bool	writeRankedNames	( string	fileName ) const;

	int		size				( void ) const;
	bool	isEmpty				( void ) const;

	void	addRankedName		( string	newName );
	int		findRankedName		( string	targetName ) const;
	int		findRank			( int		targetRank ) const;
	bool	deleteRankedName	( string	targetName  );
	bool	rerankRankedName	( string	name,	int newRank );
	int		percentLongNames	( void ) const;
	void	sortByName			( void );
	void	sortByRank			( void );

	NameRank	operator[](int index) const;

private:
	vector<NameRank> friends;

	void	swap( NameRank &a, NameRank &b );
};

//////////////////////////////////////////////
// User Interfrace definition and initialization
//////////////////////////////////////////////

class UserInterface
{
public:
	void processMenuCommands( NameRankDatabase &database );
private:
	bool processMenuCommand( NameRankDatabase &database );

	enum Command { COMMAND_READ,  COMMAND_WRITE,  COMMAND_ADD,  COMMAND_LOOKUP,  COMMAND_DELETE,
		COMMAND_RANK,  COMMAND_STAT,  COMMAND_PRINT,  COMMAND_QUIT };
	Command getCommand(void);


	void do_read	( NameRankDatabase &database );
	void do_write	( NameRankDatabase &database );
	void do_add		( NameRankDatabase &database );
	void do_lookup	( NameRankDatabase &database );
	void do_delete	( NameRankDatabase &database );
	void do_rank	( NameRankDatabase &database );
	void do_stat	( NameRankDatabase &database );
	void do_print	( NameRankDatabase &database );
	void do_quit	( NameRankDatabase &database );
};

///////////////////// tool prototypes /////////////////////////////

// general tools:
bool getBoolean(string prompt);
int getInteger(string prompt);
string getString(string prompt);
char getCharacter(string prompt);
int getNonNegativeInteger(string prompt);
int getBoundedInteger(string prompt, int lowerBound, int upperBound);
void pause(string prompt);
string getLine( string prompt );
char toLower( char c );
string toLower( string s );
void swap( string &a, string &b );
void swap( int &a, int &b );

void openIFStream( ifstream &in, string prompt );
void openOFStream( ofstream &out, string prompt );
bool getInteger( istream &in, int &input, bool flush );
bool getNonEmptyLine( istream &in, string &input );
bool getLine( istream &in, string &input );

//////////////////////////////////// Main /////////////////////////////////////////////

int main(void)
{
	UserInterface().processMenuCommands( NameRankDatabase() );

	return 0;
}


///////////////////// User Interface Implementation /////////////////////////

void UserInterface::processMenuCommands( NameRankDatabase &database )
{
	while ( processMenuCommand( database ) );
}

UserInterface::Command UserInterface::getCommand(void)
{
	while (true)
	{
		string command = toLower( getString( "Enter a command: " ) );
		if ( command == "read" )	return COMMAND_READ;
		if ( command == "write" )	return COMMAND_WRITE;
		if ( command == "add" )		return COMMAND_ADD;
		if ( command == "lookup" )	return COMMAND_LOOKUP;
		if ( command == "delete" )	return COMMAND_DELETE;
		if ( command == "rank" )	return COMMAND_RANK;
		if ( command == "stat" )	return COMMAND_STAT;
		if ( command == "print" )	return COMMAND_PRINT;
		if ( command == "quit" )	return COMMAND_QUIT;
		cout << "Valid commands are: read, write, add, lookup, delete, rank, stat, print, quit" << endl;
	}
}

bool UserInterface::processMenuCommand( NameRankDatabase &database )
{
	switch ( getCommand() )
	{
	case COMMAND_READ:		do_read(database);		return true;
	case COMMAND_WRITE:		do_write(database);		return true;
	case COMMAND_ADD:		do_add(database);		return true;
	case COMMAND_LOOKUP:	do_lookup(database);	return true;
	case COMMAND_DELETE:	do_delete(database);	return true;
	case COMMAND_RANK:		do_rank(database);		return true;
	case COMMAND_STAT:		do_stat(database);		return true;
	case COMMAND_PRINT:		do_print(database);		return true;
	case COMMAND_QUIT:		do_quit(database);		return false;
	}
}

////////////////////
// Misc. Constants
////////////////////

// option switches for file read functions
const bool FLUSH_LINE		= true;
const bool NO_FLUSH_LINE	= false;

// used for find()
const int NO_INDEX = -1;
const int NO_PERCENT = -1;

///////////////////////
// Function Prototypes
///////////////////////

void UserInterface::do_read( NameRankDatabase &database )
{
	while ( !database.readRankedNames( getString( "Enter name of data file to read: " ) ) );
}

void UserInterface::do_write( NameRankDatabase &database )
{
	while ( !database.writeRankedNames( getString( "Enter name of data file to write: " ) ) );
}

void UserInterface::do_add( NameRankDatabase &database )
{
	database.addRankedName( getString("Enter name to be added: ") );
	cout << database[database.size()-1].name << " added with rank " << database[database.size()-1].rank << '.' << endl;
}

void UserInterface::do_lookup(	NameRankDatabase &database )
{
	if ( database.size() <= 0 )
	{
		cout << "Error: There is nothing to lookup.  Operation aborted." << endl;
		return;
	}

	string	desiredName;
	int	desiredRank;

	if ( !getBoolean( "Do you want to lookup a name (otherwise you will lookup a rank)? " ) )
	{
		desiredRank = getBoundedInteger("Enter a rank to lookup: ", 1, database.size() );
		desiredName = database[ database.findRank( desiredRank ) ].name;
	}
	else
	{
		desiredName = getString( "Enter a name to lookup: " );
		int nameIndex = database.findRankedName( desiredName );
		if ( nameIndex == NO_INDEX )
		{
			cout << "Error: " << desiredName << " is not in the list.  Operation aborted." << endl;
			return;
		}
		desiredRank = database[nameIndex].rank;
	}

	cout << desiredName << " is ranked " << desiredRank << "." << endl;
}

void UserInterface::do_delete( NameRankDatabase &database )
{
	string targetName = getString( "Enter name of friend to delete: " );

	if ( database.deleteRankedName( targetName ) )
	{
		cout << targetName << " deleted." << endl;
	}
	else
	{
		cout << "Error: " << targetName << " is not in the list.  Operation aborted." << endl;
	}
}

void UserInterface::do_rank( NameRankDatabase &database )
{
	// find out whose rank to change
	string targetName = getString( "Enter name of friend to rank: " );
	int targetIndex = database.findRankedName( targetName );
	if ( targetIndex == NO_INDEX )
	{
		cout << "Error: " << targetName << " is not in the list.  Operation aborted." << endl;
		return;
	}

	// get this person's new rank
	string prompt = string("Enter new rank for ") + string(targetName) + string(": ");
	int newRank = getBoundedInteger( prompt, 1, database.size() );

	if ( database.rerankRankedName( targetName, newRank ) )
	{
		cout << targetName << " is now ranked " << newRank << '.' << endl;
	}
	else
	{
		cout << "Operation failed for unknown reason." << endl;
	}
}

void UserInterface::do_stat( NameRankDatabase &database )
{
	int statResult	= database.percentLongNames();
	int statPercent	= ( statResult == NO_PERCENT ) ? 100 : statResult;
	cout << statPercent << "% of names are long." << endl;
}

void UserInterface::do_print( NameRankDatabase &database )
{
	if ( database.isEmpty() )
	{
		cout << "The list is empty." << endl;
		return;
	}

	// determine and sort data into desired output order 
	if ( getBoolean( "Do you want the list ranked (otherwise, it will be alphabetical)? " ) )
		database.sortByRank();
	else
		database.sortByName();

	// output the data
	for ( int index = 0 ; index < database.size() ; ++index )
	{
		cout.width(4);
		cout << database[index].rank << ' ' << database[index].name << endl;
	}
}

void UserInterface::do_quit( NameRankDatabase &database )
{
	if ( getBoolean( "Save before quitting? " ) )
		do_write( database );
}

//////////////////////// NameRankDatabase implementation ///////////////////

int NameRankDatabase::size(void) const { return friends.size(); }

NameRank NameRankDatabase::operator[](int index) const
{
	return (index < 0 || index >= size() )
		? NameRank("",0) : friends[index];
}

bool NameRankDatabase::isEmpty( void ) const
{
	return friends.empty();
}

void NameRankDatabase::sortByName( void )
{
	for ( int leftBagLocation = 0 ; leftBagLocation < friends.size()-1 ; ++leftBagLocation )
	{
		int locationOfSmallestSoFar = leftBagLocation;
		for ( int search = leftBagLocation ; search < friends.size() ; ++search )
			if ( friends[search].name < friends[locationOfSmallestSoFar].name )
				locationOfSmallestSoFar = search;
		swap( friends[leftBagLocation], friends[locationOfSmallestSoFar] );
	}
}

void NameRankDatabase::sortByRank( void )
{
	for ( int leftBagLocation = 0 ; leftBagLocation < friends.size()-1 ; ++leftBagLocation )
	{
		int locationOfSmallestSoFar = leftBagLocation;
		for ( int search = leftBagLocation ; search < friends.size() ; ++search )
			if ( friends[search].rank < friends[locationOfSmallestSoFar].rank )
				locationOfSmallestSoFar = search;
		swap( friends[leftBagLocation], friends[locationOfSmallestSoFar] );
	}
}

void NameRankDatabase::addRankedName ( string newName )
{
	friends.push_back( NameRank(newName, size()+1) );
}

int  NameRankDatabase::findRankedName ( string targetName ) const
{
	for ( int searchIndex = 0 ; searchIndex < size() ; ++searchIndex )
		if ( friends[searchIndex].name == targetName )
			return searchIndex;
	return NO_INDEX;
}

int NameRankDatabase::findRank( int targetRank ) const
{
	for ( int searchIndex = 0 ; searchIndex < size() ; ++searchIndex )
		if ( friends[searchIndex].rank == targetRank )
			return searchIndex;
	return NO_INDEX;
}

bool NameRankDatabase::deleteRankedName ( string targetName )
{
	int targetIndex = findRankedName( targetName );
	if ( targetIndex == NO_INDEX ) return false;

	// adjust ranks
	int rank = friends[targetIndex].rank;
	for ( int index = 0 ; index < size() ; ++index )
		if ( friends[index].rank > rank )
			--(friends[index].rank);

	// delete item
	swap( friends[targetIndex], friends[size()-1] );
	friends.pop_back();

	return true;
}

bool NameRankDatabase::rerankRankedName ( string name, int newRank )
{
	int targetIndex = findRankedName( name );
	if ( targetIndex == NO_INDEX || newRank < 0 || newRank > size()  )
		return false;

	// do the re-ranking
	int oldRank = friends[targetIndex].rank;
	if ( newRank < oldRank )
		// add one to all friends this one is being moved in front of
		for ( int index = 0 ; index < size() ; ++index )
			if ( friends[index].rank >= newRank && friends[index].rank < oldRank )
				++friends[index].rank;
	if ( newRank > oldRank )
		// subtract one from all friends this one is being moved behind
		for ( int index = 0 ; index < size() ; ++index )
			if ( friends[index].rank > oldRank && friends[index].rank <= newRank )
				--friends[index].rank;
	friends[targetIndex].rank = newRank;

	return true;
}

bool NameRankDatabase::writeRankedNames ( string fileName ) const
{
	ofstream dataFile(fileName);
	if ( !dataFile ) return false;

	// write all of the list data to the file
	for ( int index = 0 ; index < size() ; ++index )
		dataFile << friends[ index ].name << ' ' << friends[ index ].rank << endl;

	return true;
}

bool NameRankDatabase::readRankedNames ( string fileName )
{
	// prepare to read
	ifstream dataFile( fileName.c_str() );
	if ( !dataFile ) return false;

	// empty the list.
	friends.clear();

	// fill the list from the file
	do
	{
		// attempt input
		string nameInput;
		int rankInput;
		dataFile >> nameInput >> rankInput;

		// quit if file is exhausted
		if ( !dataFile ) break;

		// attempt the add
		addRankedName( nameInput );
		friends[size()-1].rank = rankInput; // ? need this???

	} while (true);

	return true;
}

int NameRankDatabase::percentLongNames( void ) const
{
	if  ( size() <= 0 )
		return NO_PERCENT;

	int longNamesCount = 0;
	for ( int searchIndex = 0 ; searchIndex < size() ; ++searchIndex )
		if ( friends[searchIndex].name.length() > friends[searchIndex].rank )
			++longNamesCount;

	float	fractionOfTotal			= static_cast<float>(longNamesCount) / friends.size();
	float	percentOfTotal			= 100 * fractionOfTotal;
	int		roundedPercentOfTotal	= int(percentOfTotal + 0.5);
	return roundedPercentOfTotal;
}

void NameRankDatabase::swap( NameRank &a, NameRank &b )
{
	NameRank t = a;
	a = b;
	b = t;
}

///////////////////////////////////////// basic tools ///////////////////////////////////////////

bool getBoolean(string prompt)
{
	do
	{
		switch ( getCharacter(prompt) )
		{
		case 'y': case 'Y': return true;
		case 'n': case 'N': return false;
		}
		cout << "Please enter Y or N." << endl;
	} while (true);
}

int getBoundedInteger( string prompt,
						int lowerBound, int upperBound )
{
	do
	{
		int userInput = getInteger(prompt);
		if ( userInput >= lowerBound
				&& userInput <= upperBound )
			return userInput;
		cout << "Input must be in range "
			<< lowerBound << " to " << upperBound << endl;
	} while (true);
}

int getInteger(string prompt)
{
	do
	{
		int userInput;
		cout << prompt;
		cin >> userInput;
		if ( cin )
		{
			cin.ignore(999,'\n');
			return userInput;
		}
		cin.clear();
		cin.ignore(999,'\n');
		cout << "Hey, idiot!  Give me a N U M B E R!!!!" << endl;
	} while (true);
}

string getString(string prompt)
{
	do
	{
		string userInput;
		cout << prompt;
		cin >> userInput;
		if ( cin )
		{
			cin.ignore(999,'\n');
			return userInput;
		}
		cin.clear();
		cin.ignore(999,'\n');
		cout << "Hey, idiot!  Give me a string!!!!" << endl;
	} while (true);
}

char getCharacter(string prompt)
{
	do
	{
		char userInput;
		cout << prompt;
		cin >> userInput;
		if ( cin )
		{
			cin.ignore(999,'\n');
			return userInput;
		}
		cin.clear();
		cin.ignore(999,'\n');
		cout << "Input error: Please try again." << endl;
	} while (true);
}


int getNonNegativeInteger(string prompt)
{
	return getBoundedInteger(prompt, 0, INT_MAX);
}

string getLine( string prompt )
{
	do
	{
		string userInput;
		cout << prompt;
		getline(cin,userInput);
		if ( cin )
			return userInput;
		cin.clear();
		cin.ignore(999,'\n');
		cout << "Input error: Please try again." << endl;
	} while (true);
}

void pause(string prompt)
{
	cout << prompt;
	cin.ignore(999,'\n');
}

void openIFStream( ifstream &in, string prompt )
{
	do
	{
		in.open( getLine(prompt) );
		if ( in ) break;
		cout << "File Open Error: Please try again." << endl;
		in.clear();
	} while (true);
}

void openOFStream( ofstream &out, string prompt )
{
	do
	{
		out.open( getLine(prompt) );
		if ( out ) break;
		cout << "File Open Error: Please try again." << endl;
		out.clear();
	} while (true);
}

bool getLine( istream &in, string &input )
{
	getline(in,input);
	return in ? true : false;
}

bool getNonEmptyLine( istream &in, string &input )
{
	do
	{
		if ( !getLine(in,input) )
		{
			in.clear();
			return false;
		}
		if ( !input.empty() )
			return true;
	} while (true);
}

bool getInteger( istream &in, int &input, bool flush )
{
	in >> input;
	if ( !in )
	{
		in.clear();
		return false;
	}
	if (flush)
	{
		in.ignore(99,'\n');
		in.clear();
	}
	return true;
}

void swap( int &a, int &b )
{
	int t = a;
	a = b;
	b = t;
}

char toLower( char c )
{
	return tolower(c);
}
string toLower( string s )
{
	for ( int charIndex = 0 ; charIndex < s.length() ; ++charIndex )
		s[charIndex] = toLower(s[charIndex]);
	return s;
}