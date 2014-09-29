// Dynamic Counter by Phillip Ly
// This is a word counter that use dynamic implementation to generate 
// statistics (into another file) on how often words appear in a file
// It should ask the user for a file containing text/string of words 
// and count how often each word in the file occurs by outputting two
// columns of statistics. One column represent the possible words in 
// the file and the other column of statistic represents the respective
// numbers of time that the words occur in the chosen file. It should 
// then output these statistics out to a new file. The name of the new 
// file should be the name of the old file with the string “-statistics” 
// appended at the end of the base name (right before any extension like
// ".txt or .cpp”. The statistics should be outputted in decreasing order 
// of counts, but all of the words with the same counts should be outputted 
// in increasing alphabetic order.


#include<iostream> // For input and output 
#include<fstream>  // For accessing files
#include<string>   
using namespace std;

const string LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const int NPOS = -1; // For word search
const int STAT_COUNT_WIDTH = 10;

typedef char *C_String;
typedef C_String *String_Array;
typedef int *Int_Array;

//////////////////// These are the definitions for the dynammic implementation ////////////////

const int ARRAY_INITIAL_SIZE = 10;
const float ARRAY_GROWTH_RATE = 1.2f;

void construct( String_Array &strings, int &size, int &capacity );
void construct( Int_Array &ints, int &size, int &capacity );
bool isFull( String_Array &strings, int &size, int &capacity );
bool isFull( Int_Array &strings, int &size, int &capacity );
void grow( String_Array &strings, int size, int &capacity );
void grow( Int_Array &ints, int size, int &capacity );
void push_back( String_Array &strings, int &stringsSize, int &stringsCapacity,
					string aString );
void push_back( Int_Array &ints, int &intsSize, int &intsCapacity,
					int anInt );
void destruct( String_Array strings, int size, int capacity );
void destruct( Int_Array ints, int size, int capacity );

////////////////// changed application specific function prototypes //////////////

void getAndStatData(  String_Array &words, int &wordsSize, int &wordsCapacity,
						Int_Array &counts, int &countsSize, int &countsCapacity,
						string &fileName );

void sortStats( String_Array &words, int &wordsSize, int &wordsCapacity,
						Int_Array &counts, int &countsSize, int &countsCapacity );
void statWord( string word,
				String_Array &words, int &wordsSize, int &wordsCapacity,
				Int_Array &counts, int &countsSize, int &countsCapacity );

void writeStatFile( String_Array words, int wordsSize, int wordsCapacity,
					Int_Array counts, int countsSize, int countsCapacity,
					string statFileName );

int find( C_String word, String_Array words );

///////////////// unchanged application specific function prototypes ////////////////////////

string getWord( istream &inFile );
string getWord_charBased( istream &inFile ); // an alternate method of inputting words
string buildStatFileName( string fileName );
void loadFromFile( String_Array words, Int_Array counts, istream &textFile );
string popWord( string &line );
void chomp( string &line );
bool comesBefore( string word1, int count1, string word2, int count2 );

//////////////////////////////// tool prototypes //////////////////////////////////

void pause(string prompt);
string getLine( string prompt );

string toLower( string s );

string openIFStream( ifstream &in, string prompt );
void openOFStream( ofstream &out, string prompt );
bool getInteger( istream &in, int &input, bool flush );
bool getNonEmptyLine( istream &in, string &input );
bool getLine( istream &in, string &input );

////////////////////////////// application specific implementations ////////////////////////

int main(void)
{
	// set up dynamic arrays:

	String_Array words;
	int wordsSize, wordsCapacity;
	construct( words, wordsSize, wordsCapacity );

	Int_Array counts;
	int countsSize, countsCapacity;
	construct( counts, countsSize, countsCapacity );

	// get and stat the data
	string textFileName;
	getAndStatData( words, wordsSize, wordsCapacity, counts, countsSize, countsCapacity, textFileName );

	// determine output order
	sortStats( words, wordsSize, wordsCapacity, counts, countsSize, countsCapacity );

	// output results
	string statFileName = buildStatFileName( textFileName );
	writeStatFile( words, wordsSize, wordsCapacity, counts, countsSize, countsCapacity, statFileName );

	// clean up dynamic arrays
	destruct( words, wordsSize, wordsCapacity );
	destruct( counts, countsSize, countsCapacity );

	// Ending message to exit the program 
	cout << endl;
	pause("A file containing the requested statistics was created.  Press ENTER to finish...");
	return 0;

}

////////////////////////// new implementations ///////////////////////////////////

void construct( String_Array &strings, int &size, int &capacity )
{
	size = 0;
	capacity = ARRAY_INITIAL_SIZE;
	strings = new C_String[ capacity ];
}
void construct( Int_Array &ints, int &size, int &capacity )
{
	size = 0;
	capacity = ARRAY_INITIAL_SIZE;
	ints = new int[ capacity ];
}
bool isFull( String_Array &strings, int &size, int &capacity )
{
	return size >= capacity;
}
bool isFull( Int_Array &strings, int &size, int &capacity )
{
	return size >= capacity;
}
void grow( String_Array &strings, int size, int &capacity )
{
	String_Array old = strings;

	capacity = int(capacity*ARRAY_GROWTH_RATE);
	strings = new C_String[ capacity ];
	for ( int shiftIndex = 0 ; shiftIndex < size ; ++shiftIndex )
		strings[ shiftIndex ] = old[ shiftIndex ];
	delete [] old;
}
void grow( Int_Array &ints, int size, int &capacity )
{
	Int_Array old = ints;

	capacity = int(capacity*ARRAY_GROWTH_RATE);
	ints = new int[ capacity ];
	for ( int shiftIndex = 0 ; shiftIndex < size ; ++shiftIndex )
		ints[ shiftIndex ] = old[ shiftIndex ];
	delete [] old;
}
void push_back( String_Array &strings, int &stringsSize, int &stringsCapacity,
					string aString )
{
	if ( isFull( strings, stringsSize, stringsCapacity ) )
		grow( strings, stringsSize, stringsCapacity );
	strings[ stringsSize++ ] = new char[ 1 + aString.length() ];
	strcpy( strings[ stringsSize-1 ], aString.c_str() );
}
void push_back( Int_Array &ints, int &intsSize, int &intsCapacity,
					int anInt )
{
	if ( isFull( ints, intsSize, intsCapacity ) )
		grow( ints, intsSize, intsCapacity );
	ints[ intsSize++ ] = anInt;
}
void destruct( String_Array strings, int size, int capacity )
{
	for ( int deleteIndex = 0 ; deleteIndex < size ; ++deleteIndex )
		delete [] strings[deleteIndex];
	delete [] strings;
}
void destruct( Int_Array ints, int size, int capacity )
{
	delete [] ints;
}


/////////////////////////// implementations /////////////////////////////////

void writeStatFile( String_Array words, int wordsSize, int wordsCapacity,
					Int_Array counts, int countsSize, int countsCapacity,
					string statFileName )
{
	ofstream statsFile( statFileName );
	for ( int wordIndex = 0 ; wordIndex < wordsSize ; ++wordIndex )
	{
		statsFile.setf(ios::right);
		statsFile.width(STAT_COUNT_WIDTH);
		statsFile << counts[wordIndex];

		statsFile << " : ";

		statsFile << words[wordIndex] << endl;
	}
}

void loadFromFile( String_Array &words, int &wordsSize, int &wordsCapacity,
					Int_Array &counts, int &countsSize, int &countsCapacity,
					istream &textFile )
{
	do
	{
		string word = getWord( textFile );
		if ( word.length() == 0 ) break;
		statWord( toLower(word),
					words, wordsSize, wordsCapacity,
					counts, countsSize, countsCapacity );
	} while (true);
}

void chomp( string &line ) // remove leading garbage from string
{
	string::size_type firstLetter = line.find_first_of( LETTERS );
	string::size_type lengthOfLeadingGarbage = firstLetter;
	string::size_type lengthOfRemainingString = line.length() - lengthOfLeadingGarbage;
	line = ( firstLetter == string::npos ) 
				? string("") : line.substr( firstLetter, lengthOfRemainingString );
}

// get/remove first word from a string
string popWord( string &line )
{
	chomp( line );
	if ( line.length() == 0 ) return "";

	string::size_type pastWordEnd = line.find_first_not_of( LETTERS, 1 );
	string::size_type wordEnd = (pastWordEnd==string::npos)
									? (line.length()-1)
									: (pastWordEnd-1);
	string::size_type wordLength = 1 + wordEnd;
	string word = line.substr( 0, wordLength );
	line.erase( 0, wordLength );
	return word;
}

// string-based word input, inputting a line at a time
// and giving back pieces from that line as needed
// returns empty word after all words have been input
string getWord( istream &inFile )
{
	// this is static in order to keep the unprocessed input between calls
	static string inputLine = "";

	string word;
	do
	{
		word = popWord( inputLine );
		if ( word != "" ) break;

		// there are no more words in the current input line - try to get more
		getline( inFile, inputLine );
		if ( inFile.fail() ) return "";
	} while (true);

	return word;
}

// alternate getWord:
// character-based word input, treating all non-letter characters as whitespace (word delimiters)
// returns empty string after all words have been input
string getWord_charBased( istream &inFile )
{
	string word;
	do
	{
		char c = inFile.get();
		if (!inFile) return word;
		if ( isupper(c) || islower(c) )
			word += c;
		else if ( word.length() > 0 )
			return word;
	} while (true);
}

string toLower( string s )
{
	for ( unsigned i = 0 ; i < s.length() ; ++i )
		if ( isupper( s[i] ) )
			s[i] = tolower( s[i] );
	return s;
}

void swap( int &a, int &b )
{
	int temp = a;
	a = b;
	b = temp;
}

void swap( string &a, string &b )
{
	string temp = a;
	a = b;
	b = temp;
}

// does word1/count1 come before word2/count in desired output ordering?
bool comesBefore( string word1, int count1, string word2, int count2 )
{
	if ( count1 > count2 ) return true;		// larger counts first
	if ( count1 < count2 ) return false;	// smaller counts later
	return word1 < word2;					// same counts: use word order
}
// a cleaner, although somewhat more opaque, way to test order:
bool comesBefore2( string word1, int count1, string word2, int count2 )
{
	if ( count1 != count2 )
		return count1 > count2;		// order by (decreasing) count if counts are different
		return word1 < word2;		// order by word if counts are the same
}

int findFirst( Int_Array counts, int countsSize, int countsCapacity,
				String_Array words, int wordsSize, int wordsCapacity,
				int startSearch )
{
	// order by decreasing count
	// then (when the counts are equal) by increasing alphabetical (not required, but nice)
	int locationOfBestSoFar = startSearch;
	for ( int search = startSearch+1 ; search < countsSize ; ++search )
		if ( comesBefore( string(words[search]), counts[search],
							string(words[locationOfBestSoFar]), counts[locationOfBestSoFar] ) )
			locationOfBestSoFar = search;
	return locationOfBestSoFar;
}

void sortStats( String_Array &words, int &wordsSize, int &wordsCapacity,
				Int_Array &counts, int &countsSize, int &countsCapacity )
{
	for ( int locationToSelect = 0 ; locationToSelect < wordsSize ; ++locationToSelect )
	{
		int locationToSwap = findFirst( counts, countsSize, countsCapacity,
										words, wordsSize, wordsCapacity,
										locationToSelect );
		swap( words[locationToSelect], words[locationToSwap] );
		swap( counts[locationToSelect], counts[locationToSwap] );
	}
}

int find( string word, String_Array words, int wordsSize, int wordsCapacity )
{
	for ( int search = 0 ; search < wordsSize ; ++search )
		if ( word == words[search] )
			return search;
	return NPOS;
}

void statWord( string word,
				String_Array &words, int &wordsSize, int &wordsCapacity,
				Int_Array &counts, int &countsSize, int &countsCapacity )
{
	int location = find( word, words, wordsSize, wordsCapacity );
	if ( location != NPOS )
	{
		// add to existing count for this word
		counts[location]++;
	}
	else
	{
		// add word to list
		push_back(words, wordsSize, wordsCapacity, word);
		push_back(counts, countsSize, countsCapacity, 1);
	}
}
// This function manipulate the resulted file's name by adding "-statistics" to the base name
// of the file that the user entered
string buildStatFileName( string fileName )
{
	string::size_type ext = fileName.find_last_of( "." );
	string baseName = ( ext == string::npos ) ? fileName : fileName.substr(0,ext);
	string extension = ( ext == string::npos ) ? "" : fileName.substr(ext, fileName.length()+1-ext);
	return baseName + "-statistics" + extension;
}

void getAndStatData(  String_Array &words, int &wordsSize, int &wordsCapacity,
						Int_Array &counts, int &countsSize, int &countsCapacity,
						string &fileName )
{
	ifstream textFile;
	fileName = openIFStream( textFile, "Enter name of file containing text: " );
	loadFromFile( words, wordsSize, wordsCapacity,
					counts, countsSize, countsCapacity,
					textFile );
	textFile.close();
}

/////////////////////////// tool implimentations ////////////////////////////////

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

string openIFStream( ifstream &in, string prompt )
{
	do
	{
		string fileName = getLine(prompt);
		in.open( fileName );
		if ( in ) return fileName;
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

