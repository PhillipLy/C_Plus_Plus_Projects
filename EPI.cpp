//Elements of Programming Problems

//////////////////////////////////////////////////////
Chapter 5 Primitive Types
//////////////////////////////////////////////////////

Boot camp problem

Write a program to count the number of bits that are set
to 1 in an unsigned integer.

short CountBits( unsigned int x ) {
	short num_bits = 0;
	while (x) {
		num_bits += x & 1;
		x >>= 1;
	}
	return num_bits;
}
// Time: O(n), where n is the number of bits in integer word


Problem:  you need to know the unsigned integer and it's binary format, 
retrieve the substring of the binary format .


#include <iostream>
#include <limits>
#include <string>

using namespace std;

string toBinStr(unsigned int val)
{
	// mask has only the leftmost bit set.
	unsigned int mask = 1u << (std::numeric_limits<unsigned>::digits-1) ;

	// skip leading bits that are not set.
	while ( 0 == (val & mask) && mask != 0 )
		mask >>= 1 ; // shift all bits to the right by 1

	string binStr ;
	binStr.reserve(std::numeric_limits<unsigned>::digits+1) ;

	do
	{
		// add a '1' or '0' depending the current bit.
		binStr += static_cast<char>(val & mask) + '0' ;

	} while ( (mask>>=1) != 0 ) ; // next bit, when mask is 0 we've processed all bits

	return binStr ;
}

int main()
{
   for(unsigned int i = 0; i < 128; i++)
   {
      string entity = toBinStr(i) ;
      cout << entity << '\n' ;
   }
}


5.1 Computing the parity of a word
The parity of a binary word is 1 if the number of 1s in the word
is odd; otherwise, it is 0. How to compute the parity of a very
large number of 64-bit words?

short Parity( unsigned long x ) {
	short result = 0;
	while (x) {
		result ^= 1;
		x &= (x - 1);
	}
	return result;
}
// Time: O( k )

short Parity( unsigned long x ) {
	
	x ^= x >> 32;
	x ^= x >> 16;
	x ^= x >> 8;
	x ^= x >> 4;
	x ^= x >> 2;
	x ^= x >> 1;
	
	return x & 0x1;	
}
// Time: O( log n ), where n is the word size



5.5 Compute X x Y without arithmetical operators

unsigned Multiply( unsigned x, unsigned y ) {
	
	unsigned sum = 0;
	while (x) {
		if (x & 1) {
			sum = Add(sum, y);
		}
		x >>= 1, y <<= 1;
	}
	return sum;
}

unsigned Add (unsigned a, unsigned b) {
	
	unsigned sum = 0, carryin = 0, k = 1, temp_a = a, temp_b = b;
	
	while (temp_a || temp_b) {
		unsigned ak = a & k, bk = b & k;
		unsigned carryout = (ak & bk) | (ak & carryin) | (bk & carryin));
		sum |= (ak ^ bk ^ carryin);
		carryin = carryout << 1, k <<= 1, temp_a >>= 1, temp_b >>= 1;
	}
	return sum | carryin;
}
// Time: O( n ), where n is the width of operand


5.8 Reverse Digits

Reverse of 42 is 24. Reverse of 314 is 413.

long Reverse( int x ) {
	
	long result = 0, x_remaining = abs(x);
	while (x_remaining) {
		result = result * 10 + x_remaining % 10;
		x_remaining /= 10;
	}
	
	return x < 0 ? -result : result;	
}
// Time: O(n) 



5.11 Rectangle Intersection

Write a program to test if two rectangles have a 
nonempty intersection. If the intersection is nonempty,
return the rectangle formed by their intersection.

struct Rectangle {
	int x, y, width, height;
};

Rectangle IntersectionRectangle( const Rectangle & R1, const Rectangle &R2) {
	
	if (!IsIntersect(R1,R2)) {
		return {0, 0, -1, -1}; // No intersection
	}
	return { max(R1.x, R2.x), max(R1.y, R2.y),
			 min(R1.x + R1.width, R2.x + R2.width) - max(R1.x, R2.x),
	         min(R1.y + R1.height, R2.y + R2.height) - max (R1.y, R2.y)};
			 
}

bool IsIntersect (const Rectangle &R1, const Rectangle &R2) {
	
	return R1.x <= R2.x + R2.width && R1.x + R1.width >= R2.x &&
		   R1.y <= R2.y + R2.width && R1.y + R1.height >= R2.y;
}

// Time complexity is O(1) because the number of 
// operations is constant.



///////////////////////////////////////////////////////
Chapter 6 Array
///////////////////////////////////////////////////////

Retrieving and update A[i] takes O(1) time.
Deleting an element at index i from an array of length
n is O(n - i).

// Key methods: 
array<int,3> A = {1, 2, 3};
vector<int> A = {1, 2, 3};

// Construct a subarray from an array
vector<int> subarray_A(A.begin() + i, A.begin() + j);

// Instantiate 2D Array
array<array<int,2>,3> A = { {1, 2}, {3, 4}, {5, 6} }
vector<vector<int>> A = { {1, 2}, {3, 4} };

// Add values to end of dynamically sizable vector
push_back(42);
emplace_back(42);

//important array methods:
binary_search(A.begin(), A.end(), 42);
lower_bound(A.begin(), A.end(), 42);
upper_bound(A.begin(), A.end(), 42);
max_element(A.begin(), A.end());
reverse(A.begin(), A.end());
rotate(A.begin(), A.begin() + shift, A.end());

sort(A.begin(), A.end());



Array Bootcamp: even or odd

void EvenOdd( vector<int> * A_ptr ) {
	
	vector<int> &A = *A_ptr;
	int next_even = 0; 
	int next_odd = A.size() - 1;
	
	while (next_even < next_odd) {
		if ( A[next_even] % 2 == 0 ) {
			++next_even;
		}
		else {
			swap(A[next_even], A[next_odd--]);
		}
	}
}
// Time: O(n)


6.1 The Dutch National Flag

Write a program that takes an array A and an index i
into A, and rearranges the elements such that all 
elements less A[i] (the "pivot") appear first, followed
by elements equal to the pivot, followed by elements 
greater than the pivot.

// method 1

typedef enum { RED, WHITE, BLUE } Color;

void DutchFlagPartition( int pivot_index, vector<Color> *A_ptr ) {
	
	vector<Color> &A = *A_ptr;
	Color pivot = A[pivot_index];
	
	// First pass: group elements smaller than pivot
	int smaller = 0;
	for (int i = 0; i < A.size(); ++i) {
		if (A[i] < pivot) {
			swap(A[i], A[smaller++]);
		}
	}
	
	int larger = A.size() - 1;
	for (int i = A.size() - 1; i >= 0 && A[i] >= pivot; --i) {
		if (A[i] > pivot) {
			swap(A[i], A[larger--]);
		}		
	}	
}

// Time complexity is O(n). Space complexity: O(1)


// method 2

typedef enum { RED, WHITE, BLUE } Color;

void DutchFlagPartition( int pivot_index, vector<Color> * A_ptr ) {
	
	vector<Color> &A = *A_ptr;
	Color pivot = A[pivot_index];
	
	int smaller = 0, equal = 0, larger = A.size();
	// Keep iterating as long as there is an unclassified element
	while (equal < larger) {
		if (A[equal] < pivot) {
			swap(A[smaller++], A[equal++]);
		}
		else if (A[equal] == pivot) {
			++equal;
		}
		else { // A[equal] > pivot
			swap(A[equal], A[--larger]);
		}
	}
}
//Time complexity: O(n). Space complexity: O(1)

6.2 Increment an arbitrary-precision integer
Write a program which takes as input an array of digits 
encoding a decimal number D and updates the array to 
represent the number D + 1. For example, if the input is
<1, 2, 9> then the ouput is <1, 3, 0>

vector<int> PlusOne( vector<int> A ) 
{
	++A.back() // increment last digit in array A
	
	for (int i = A.size() - 1; i > 0 && A[i] == 10; --i) {
		A[i] = 0, ++A[i - 1];
	}
	if (A[0] == 10) { // if least significant digit = 10, 
		A[0] = 0;    // A[0] is zero and  
		A.insert(A.begin(), 1) // insert 1 to the beginning 
	}	
}
// Time: O(n), where n is the length of A



6.4 Advancing through an array

Write a program which takes an array of n integers,
where A[i] denotes the maximum you can advance from
index i, and returns where it is possible to advance
to the last index starting from the beginning of the
array.

bool CanReachEnd( const vector<int> &max_advance_steps) {
	// Instantiate variables
	int furthest_reach_so_far = 0;
	int last_index = max_steps.size() - 1;
	
	for (int i = 0; i <= furthest_reach_so_far && furthest_reach_so_far < last_index; ++i) {
		furthest_reach_so_far = max(furthest_reach_so_far, max_advance_steps[i] + i);
	}
	return furthest_reach_so_far >= last_index;	
}

// Time Complexity: O(n). Space Complexity is O(1)



6.5 Delete Duplicates from a sorted array

Write a program which takes as input a sorted array
and updates it so that all duplicates have been removed
and the remaining elements have been shifted to left to
fill the emptied indices. Return the number of valid
elements.

int DeleteDuplicates(vector<int> *A_ptr) {
	
	vector<int> &A = *A_ptr; //instantiate vector A_ptr
	// base case
	if (A.empty()) {
		return 0;
	}
	
	int write_index = 1;
	
	for (int = 1; i < A.size(); ++i) {
		if (A[write_index - 1] != A[i]) {
			A[write_index++] = A[i];
		}
	}	
	return write_index;
}

// Time complexity: O(n). Space complexity: O(1).


6.6 Buy and Sell A stock once

Write a program that takes an array denoting the daily
stock price, and returns the maximum profit that could
be made by buying and then selling one share of that
stock.

double BuyAndSellStockOnce( const vector<double> & prices ) {
	
	double min_price_so_far = numeric_limits<double>::max(), max_profit = 0;

	for (const double &price : prices) {
		double max_profit_sell_today = price - min_price_so_far;
			   max_profit = max(max_profit, max_profit_sell_today);
			   min_price_so_far = min(min_price_so_far, price);
	}
	return max_profit;
}

// Time complexity: O(n). Space complexity: O(1).


6.8 Enumerate all primes to name

A natural numer is called a prime if it is bigger than
1 and has no divisors other than 1 and itself

Write a program that takes an integer argument and
returns all the primes between 1 and that integer.
For example, if the input is 18, you should return
<2, 3, 5, 7, 11, 13, 17>.

// Given n, return all primes up to and including n

vector<int> GeneratePrimes( int n ) {
	// instantiate primes vector
	vector<int> primes;
	deque<bool> is_prime(n + 1, true);
	is_prime[0] = is_prime[1] = false;
	
	for (int p = 2; p < n; ++p) {
		if (is_prime[p]) {
			primes.emplace_back(p);
			
			for (int j = p; j <= n; j += p) {
				is_prime[j] = false;
			}
		}
	}
	return primes;
}

// Time complexity: O(nloglogn). Space complexity: O(n)


6.9 Permute the elements of an array

Given an array A of n elements and permutation P,
apply P to A

void ApplyPermutation(vector<int> *perm_ptr, vector<int> * A_ptr) {
	
	//instantiate vectors for use
	vector<int> &perm = *perm_ptr;
	vector<int> &A = *A_ptr;
	
	for (int i = 0; i < A.size(); ++i) {
		//check if the element at index i has not been
		// moved by checking if perm[i] is nonnegative
		int next = i;
		while (perm[next] >= 0) {
			swap(A[i], A[perm[next]]);
			int temp = perm[next];
			// substracts perm.size() from an entry in
			// perm to make it negative
			perm[next] -= perm.size();
			next = temp;
		}
	}
	// Restore perm
	for_each(begin(perm), end(perm), [&perm](int& x) { x += perm.size(); });
}
// Time complexity: O(n). Space complexity: O(1)


////////////////////////////////////////////////////// 
Chapter 7: Strings
//////////////////////////////////////////////////////

// Strings boot camp

// Check for palindromic string

bool IsPalindromic( const string &s ) {
	
	for (int i = 0, j = s.size() - 1; i < j; ++i, --j) {
		if (s[i] != s[j]) {
			return false;
		}
	}
	return true;
}
// Time complexity: O(n). Space complexity: O(1)


// Strings manipulation methods
append("Phillip");
push_back('c');
pop_back();
insert(s.begin() + shift, "Phillip");
substr(pos, len);
compare("Phillip");

// Remember a string is organized like an array. It
// Performs well for operations from the back such as
// push_back('c') and pop_back(), but poorly in the
// middle of a string such as
// insert(A.begin() + middle, "Phillip")

7.3 Compute Spreadsheet column encoding

int SSDecodeColID( const string &col ) {
	
	int result = 0;
	for (char c : col) {
		result = result * 26 + c - 'A' + 1;
	}
	return result;
}
// Time complexity: O(n).


7.4 Replace and Remove

Write a program which takes as input an array of 
characters, and removes each 'b' and replaces each
'a' by two 'd's. 

int ReplaceAndRemove( int size, char s[] ) {
	
	// Forward iteration: remove "b"s and count the
	// count the number of "a"s
	int write_idx = 0;
	int a_count = 0;
	
	for (int i = 0; i < size; ++i) {
		if (s[i] != 'b') {
			s[write_idx++] = s[i];
		}
		if (s[i] == 'a') {
			++a_count;
		}
	}
	
	// Backward iteration: replace "a"s with "dd"s
	// starting from the end.
	int cur_idx = write_idx - 1;
	write_idx = write_idx + a_count - 1;
	const int final_size = write_idx + 1;
	
	while (cur_idx >= 0) {
		if (s[cur_idx] == 'a') {
			s[write_idx--] == 'd';
			s[write_idx--] == 'd';
		}
		else {
			s[write_idx--] = s[cur_idx];
		}
		--cur_idx;
	}
	return final_size;
}
// Time complexity: O(n). 


7.5 Test Palindromicity
// hint: use two indices.

Implement a function which takes as input a string s and
return true if s is a palindromic string

bool IsPalindrome( const string & s) {
	// Instantiate two indices
	int i = 0;
	int j = s.size() - 1;
	
	while (i < j ) {
		//forward
		while ( !isalnum(s[i]) && i < j ) {
			++i;
		}
		//backward
		while ( !isalnum(s[i]) && i < j ) {
			--j;
		}
		//check if two indices are equal
		if ( tolower(s[i++]) != tolower(s[j--]) ){
			return false;
		}
	}
	return true;
}
// Time complexity: O(n), where n is the length of s
 

7.6 Reverse all words in a sentence

Implement a function to reverse the words in a string s
// Hint:
"ram is costly" reverse whole string "yltsoc si mar". 
Then we reverse individual word: "costly is ram"

void ReverseWords( string *s ) {
	// Reverse the whole string first
	reverse( s->begin(), s->end() );
	
	size_t start = 0;
	size_t end;
	
	while ((end = s->find(" ", start)) != string::npos) {
		//Reverses each word in the string
		reverse( s->begin() + start, s->begin() + end );
		start = end + 1;
	}
	// Reverses the last word
	reverse( s->begin() + start, s->end() );
}
// Time complexity is O(n). Space complexity O(n).


7.12 Implement run length encoding

Implement run length encoding and decoding functions. For example,
the RLE of "aaaabcccaa" is "4a1b3c2a". The decoding of "3e4f2e"
returns "eeeffffee".

string Decoding ( const string & s ) 
{
	int count = 0; // initialize count to zero
	string result; // create variable of type string to store result
	
	for (const char &c : s) {
		if (isdigit(c)) {
			count = count * 10 + c - '0';
		}
		else { // c is a letter of alphabet
			result.append(count, c); // Appends count copies of c to result
			count = 0;
		}
	}
	return result;
}

string Encoding( const string &s ) {
	// create variable
	string result;
	
	for ( int i = 1, count = 1; i <= s.size(); ++i ) {
		if (i == s.size() || s[i] != s[i - 1]) {
			// found new character so write the count of previous char
			result += to_string(count) + s[i - 1];
			count = 1;
		}
		else { // s[i] == s[i - 1]
			++count;
		}
	}
	return result;
}
// Time: O(n)


7.13 Find the first occurence of a substring

Given two strings s(the"search string") and t(the "text")
find the first occurence of s in t
// Three linear string matching algorithms: KMP, 
// Boyer-Moore, and Rabin-Karp

//Returns the index of the first character of the
//substring if found, -1 otherwise

int RabinKarp( const string &t, const string &s ) {
	
	if (s.size() > t.size()) {
		return -1; // s is not substring of t
	}
	
	const int kBase = 26;
	int t_hash = 0, s_hash = 0; // Hash codes for t and s
	int power_s = 1; 
	
	for (int i = 0; i < s.size(); ++i) {
		power_s = i ? power_s * kBase : 1;
		t_hash = t_hash * kBase + t[i];
		s_hash = s_hash * kBase + s[i];	
	}
	
	for (int i = s.size(); i < t.size(); ++i) {
		// Checks the two substrings are equal or not,
		// to prevent hash collision
		if (t_hash == s_hash && !t.compare(i - s.size(), s.size(), s)) {
			return i - s.size(); // found a match
		}
		
		// Uses rolling hash to compute the new hash codes
		t_hash -= t[i - s.size()] * power_s;
		t_hash = t_hash * kBase + t[i];	
	}
	
	// Tries t match s and t[t.size() - s.size() : t.size() - 1]
	if (t_hash == s_hash && t.compare(t.size() - s.size(), s.size(), s) == 0) {
		return t.size() - s.size();
	}
	return -1;
}
// Time Complexity: O(m + n)


/////////////////////////////////////////////////////////
Chapter 8: Linked Lists
/////////////////////////////////////////////////////////

// Linked lists boot camp

template <typename T>
struct ListNode {
	T data;
	shared_ptr<ListNode<T>> next;
}

// Search for a key
share_ptr<ListNode<int>> SearchList(shared_ptr<ListNode<int>> L, int key) {
	
	while (L && L->data != key) {
		L = L->next;
	}
	// If key was not present in list, L is null
	return L;
}

// Insert a new node after a specified node
void InsertAfter( const shared_ptr<ListNode<int>> &node,
				  const shared_ptr<ListNode<int>> &new_node) {
	new_node->next = node->next;
	node->next = new_node;	
}

//Delete a node:
void DeleteAfter( const shared_ptr<ListNode<int>> &node ) {
	node->next = node->next->next;
}

// Insert and delete is O(1) for Linked lists
// Search is O(n), where n is the number of nodes

// Linked list methods (singly-linked lists):

// insert or delete elements in list
push_front(42);
pop_front();
insert_after(L.end(), 42)
erase_after(A.end());

// transfer elements from list to another 
splice_after(L1.end(), L2)

reverse(); // reverse list
sort(); // sort list!!!



8.1 Merge two sorted lists

Write a program that takes two lists (sorted), and 
returns their merge. 

shared_ptr<ListNode<int>> MergeTwoSortedLists(shared_ptr<ListNode<int>> L1,
											  shared_ptr<ListNode<int>> L2) {
	// Creates a placeholder for the result
	shared_ptr<ListNode<int>> dummy_head(new ListNode<int>);
	auto tail = dummy_head;
	
	while (L1 && L2) {
		AppendNode(L1->data <= L2->data ? &L1 : &L2, &tail);	
	}
	// Appends the remaining nodes of L1 or L2
	tail->next = L1 ? L1 : L2;
	return dummy_head->next;	
}

void AppendNode( shared_ptr<ListNode<int>> *node,
			     shared_ptr<ListNode<int>> *tail) {
		
	(*tail)->next = *node; 
	*tail = *node;
	*node = (*node)->next;
}
// Time complexity is O(n + m). Space complexity: O(1)



8.2 Reverse a single sublist 

Write a program which takes a singly linked list L and two 
integers s and f as arguments, and reverses the order of the
nodes from sth node to fth node, inclusive. 

shared_ptr<ListNode<int>> ReverseSublist( shared_ptr<ListNode<int> L,
										 int start, int finish ) {
	if ( start == finish) { // No need to reverse, start == finish
		return L;
	}
	
	auto dummy_head == make_shared<ListNode<int>>(ListNode<int>{0, L});
	auto sublist_head = dummy_head;
	int k = 1;
	while (k++ < start) {
		sublist_head = sublist_head->next;
	}
	
	// Reverses sublist
	auto sublist_iter = sublist_head->next;
	while ( start++ < finish ) {
		auto temp = sublist_iter->next;
		sublist_iter->next = temp->next;
		temp->next = sublist_head->next;
		sublist_head->next = temp;
	}
	return dummy_head->next;	
}
// Time: O(f), dominated by the search for fth node




8.3 Test for Cyclicity

Write a program that takes the head of a singly linked
list and returns null if there does not exist a cycle, 
and the node at the start of the cycle, if a cycle is 
present.

// Method 1

shared_ptr<ListNode<int>> HasCycle(const shared_ptr<ListNode<int>> &head) {
	
	shared_ptr<ListNode<int>> fast = head, slow = head;
	
	while (fast && fast->next && fast->next->next) {
		// slow pointer moves one step, fast pointer moves two steps
		slow = slow->next, fast = fast->next->next;
		
		if (slow == fast) { // if there is a cycle
			slow = head;
			// Both pointers advance at the same time
			while (slow != fast) {
				slow = slow->next, fast = fast->next;
			}
			return slow; // slow is the start of cycle
		}	
	}
	return nullptr; // No cycle
}
// Time: O(F) + O(C) = O(n), where n is total number of nodes


8.4 Test for and merge overlapping lists - lists are cycle-free

Write a program that takes two cycle-free singly linked lists,
and determines if there exists a node that is common to both lists

shared_ptr<ListNode<int>> OverlappingNoCycleLists (
	shared_ptr<ListNode<int>> L1, shared_ptr<ListNode<int>> L2) {
	// Create variables 
	int L1_len = Length(L1);
	int L2_len = Length(L2);
	
	// Advances the longer list to get equal length lists
	AdvanceListByK(abs(L1_len - L2_len), L1_len > L2_len ? &L1 : &L2);
	
	while (L1 && L2 && L1 != L2) {
		L1 = L1->next, L2 = L2->next;
	}
	return L1; // nullptr implies there is no overlap between L1 and L2
	
	int Length( shared_ptr<ListNode<int>> L ) {
		int length = 0;
		while ( L ) {
			++length, L = L->next;
		}
		return length;
	}	
	// Advances L by k steps
	void AdvanceListByK( int k, shared_ptr<ListNode<int>> * L ) {
		while ( k-- ) {
			*L = (*L)->next;
		}
	}	
}
// Time: O(n)  Space: O(1)



8.6 Delete a node from a singly linked list

Write a program which deletes a node in a singly linked list.
The input node is guaranteed not to be the tail node.

void DeletionFromList( const shared_ptr<ListNode<int>> & node_to_delete ) {
	node_to_delete->data = node_to_delete->next->data;
	node_to_delete->next = node_to_delete->next->next;
}
// Time: O(1)


8.8 Remove duplicates from a sorted list

Write a program that takes as input a singly linked list
of integers in sorted order, and removes duplicates from
it. The list should be sorted.

shared_ptr<ListNode<int>> RemoveDuplicates (
	const shared_ptr<ListNode<int>> & L ) {
	
	auto iter = L;
	while ( iter ) {
		auto next_distinct = iter->next;
		while (next_distinct && next_distinct->data == iter->data) {
			next_distinct = next_distinct->next;
		}
		iter->next = next_distinct;
		iter = next_distinct;
	}
	return L;
}
// Time: O(n)  Space: O(1)



8.10 Implement even odd merge singly linked list

Write a program that computes the even odd merge.

shared_ptr<ListNode<int>> EvenOddMerge( 
				const shared_ptr<ListNode<int>> & L ) {
	if ( L == nullptr ) {
		return L;
	}

	auto even_dummy_head = make_shared<ListNode<int>>(ListNode<int>{0, nullptr}),
		odd_dummy_head = make_share<ListNode<int>>(ListNode<int>{0, nullptr});
		array<shared_ptr<ListNode<int>>, 2> tails = { even_dummy_head,
													 odd_dummy_head };
	int turn = 0;
	for (auto iter = L; iter; iter = iter->next) {
		tails[turn]->next = iter;
		tails[turn] = tails[turn]->next;
		turn ^= 1; // alternate between even and odd
	}
	tails[1]->next = nullptr;
	tails[0]->next = odd_dummy_head->next;
	return even_dummy_head->next;
}
// Time: O(n)  Space: O(1)



8.11 Test whether a singly linked list is palindromic

Write a program that tests whether a singly linked list is
palindromic.

bool IsLinkedListAPalindrome( shared_ptr<ListNode<int>> L ) {
	// base case
	if ( L == nullptr ) {
		return true;
	}
	// finds the second half of L-
	shared_ptr<ListNode<int>> slow = L, fast = L;
	
	while (fast && fast->next) { 
		fast = fast->next->next;
		slow = slow->next;
	}
	// compares the first half and the reversed second half lists
	auto first_half_iter = L, second_half_iter = ReverseLinkedList(slow);
	
	while (second_half_iter && first_half_iter) {
		if ( second_half_iter->data != first_half_iter->data ) {
			return false;
		}
		second_half_iter = second_half_iter->next; // move on to next one
		first_half_iter = first_half_iter->next;
	}
	return true;
}
// Time: O(n)  Space: O(1)









///////////////////////////////////////////////////////
Chapter 9: Stacks and Queues
///////////////////////////////////////////////////////

// Stack is O(1) for push and pop
// Stack is Last-in, First-out (LIFO)

Stack libraries
push(e) pushes an element onto the stack. 
top() retrieve an element from top of stack but does not remove
pop() will remove an element at top of stack
empty() tests if the stack is empty

Boot campt problem:

Write a program that uses a stack to print the entries of
a singly linked list in reverse order.

// Method 1
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (head == nullptr) {
            return nullptr;
        }
        
        ListNode *pre = nullptr, *next = nullptr;
        
        while (head != nullptr) {
            next = head->next;
            head->next = pre;
            pre  = head;
            head = next;
        }
        
        return pre;
    }
};
// Time: O(n)  Space: O(1)

// Method 2 Recursion from discuss board!

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        return help(head, NULL);
    }
    ListNode* help(ListNode* cur, ListNode* pre){
        if (!cur) return cur;
        ListNode* next = cur->next;
        cur->next = pre;
        return !next ? cur : help(next, cur);
    }
};
// Time: O(n)  Space: O(1)


/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */

// method 3
void PrintLinkedListInReverse(shared_ptr<ListNode<int>> head) {
	stack<int> nodes;
	
	while (head) {
		nodes.push(head->data);
		head = head->next;
	}
	while (!nodes.empty()) {
		cout << nodes.top() << endl;
		nodes.pop();
	}	
}
// Time: O(n)  Space: O(n) 

// Recursion: // from discuss board!



/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */


	







9.1 Implement a stack with MAX API

Design a stack that includes a max operation, in 
addition to push and pop. The max method should return
the maximum value in the stack.

class Stack {
 public:
	bool (Empty() const { return max_element.empty(); }
	
	// Max function
	int Max() const {
		// base case for empty stack
		if (Empty()) {
			throw length_error("Max(): empty stack");
		}
		// return max element at top of stack
		return max_element.top().max;
	}
	
	// Pop function
	int Pop() {
		// Base case 
		if (Empty()) {
			throw length_error("Pop(): empty stack");
		}
		// pop element at top of stack
		int pop_element = max_element.top().element;
		max_element.pop();
		return pop_element;
	} 
	
	// Push function
	void Push (int x) {
		max_element.emplace(
		  MaxElementCached{x, max(x, Empty() ? x : Max())});
	}
	
 private: 
	struct MaxElementCached {
		int element, max;
	};
	stack<MaxElementCached> max_element;
};
// Time complexity O(1). Space complexity O(n).



9.2 Evaluate RPN Expressions

Write a program that takes an arithmetical expression in RPN
and returns the number that the expression evaluates to.

int Eval( const string & RPN_expression ) {
	// declare variables
	stack<int> intermediate_results;
	stringstream ss(RPN_expression);
	string token;
	const char kDelimiter = ',';
	
	while (getline( ss, token, kDelimiter )) {
		if (token == "+" || token == "-" || "*" || token == "/") {
			const int y = intermediate_results.top();
			intermediate_results.pop();
			const int x = intermediate_results.top();
			intermediate_results.pop();			
			switch (token.front()) {
				case '+':
					intermediate_results.emplace(x + y);
					break;
				case '-':
					intermediate_results.emplace(x - y);
					break;
				case '*':
					intermediate_results.emplace(x * y);
					break;
				case '/':
					intermediate_results.emplace(x / y);
					break;
			}
		}
		else {
			intermediate_results.emplace(stoi(token));
		}
	}
	return intermediate_results.top();	
}
// Time: O(n) 





9.3 Test a string for well-formedness

Write a program that tests if a string made up of
characters '(', ')', '[', ']', '{', '}' is well-formed

bool IsWellFormed( const string &s) {
	
	stack<char> left_chars;
	for (int i = 0; i < s.size(); ++i) {
		if (s[i] == '(' || s[i] == '{' || s[i] == '[') {
			left_chars.emplace(s[i]); //put on stack
		}
		else {
			if (left_chars.empty()) {
				return false; // unmatched right char empty case
			}
			if ((s[i] == ')' && left_chars.top() != '(') ||
		        (s[i] == '}' && left_chars.top() != '{') ||
				(s[i] == ']' && left_chars.top() != '[' )) {
				
				return false; // Mismatched chars
			}
			left_chars.pop();
		}
	}
	return left_chars.empty();
}
// Time complexity: O(n)



9.4 Normalize pathnames

Write a program which takes a pathname, and returns the shortest
equivalent pathname. Assume individual directories and files have
names that use only alphanumeric characters. Subdirectory names may
be combined using forward slashes (/), the current director(.), and
parent director(..).

string ShortestEquivalentPath( const string & path) {
	// base case
	if (path.empty()) {
		throw invalid_argument("Empty string is not a valid path.");
	}
	vector<string> path_names; // Uses vector as a stack
	if (path.front() == '/') {
		path_names.emplace_back("/");
	}
	
	stringstream ss(path);
	string token;
	while (getline(ss, token, '/')) {
		if (token == "..") {
			if (path_names.empty() || path_names.back() == "..") {
				path_names.emplace_back(token);
			}
			else {
				if (path_names.back() == "/") {
					throw invalid_argument("Path error");
				}
			}
			path_names.pop_back();
		}
		else if (token != "." && token != "") { // Must be a name
			path_names.emplace_back(token);
		}
	}
	if (!path_names.empty()) {
		result = path_names.front();
		for (int i = 1; i < path_names.size(); ++i) {
			if (i == 1 && result == "/" {
				result += path_names[i];
			}
			else {
				results += "/" + path_names[i];
			}
		}
	}
	return result;
}

// Time: O(n), where n is the length of the pathname





9.5 Search a postings list

Write recursive and iterative routines that take a postings
list, and compute the jump-first order.

// Iterative
void SetJumpOrder( const shared_ptr<PostingListNode> & L ) 
{
	stack<shared_ptr<PostingListNode>> s;
	int order = 0; 
	s.emplace(L);
	
	while ( !s.empty() ) {
		auto curr = s.top();
		s.pop();
		
		if ( curr && curr->order == -1 ) {
			curr->order = order++;
			// Stack is LIFO so push next, then push jump
			s.emplace(curr->next);
			s.emplace(curr->jump);
		}
	}	
}
// Time: O(n)  Space: O(n)






//////////////////////////////////////////////////////
// Queue section of Chapter 9

//enqueue and dequeue
// Queue boot camp FIFO- First In First Out

// Methods:
push(e);
front();
pop();

Implement basic queue API

class Queue {
 public:
	void Enqueue( int x ) { data_.emplace_back(x); }
	
	// Dequeue function
	int Dequeue() {
		if (data_.empty()) {
			throw length_error("empty queue");
		}
		//retrieve value front of queue
		const int val = data_.front();
		data_.pop_front(); // pop value front of queue
		return val;
	}
	
	// Max function
	int Max() const {
		if (data_.empty()) {
			throw length_error("Error: Max() on empty queue.").
		}
		return *max_element(data_.begin(), data_.end());
	}
	
 private: 
	list<int> data_;
	
};
// Time Complexity: O(n), where n is number of entries


9.7 Compute Binary Tree Nodes in order of increasing 
depth

Given a binary tree, return an array consisting of keys
at the same level. Example: return 
<<314>, <6,6>,<271,561, 2,271>,<28,0,3,1,28>>
// 2-d vector


vector<vector<int>> BinaryTreeDepthOrder(
	 const unique_ptr<BinaryTreeNode<int>> &tree) {
	
	queue<BinaryTreeNode<int>*> curr_depth_nodes({tree.get()});
	vector<vector<int>> result;
	
	while (!curr_depth_node.empty()) {
		queue<BinaryTreeNode<int>*> next_depth_nodes;
		vector<int> this_level;
		
		while (!curr_depth_nodes.empty()) {
			auto curr = curr_depth_node.front();
			curr_depth_node.pop();
			if (curr) {
				this_level.emplace_back(curr->data);
				
				// Defer the null checks to the null test above
				next_depth_nodes.emplace(curr->left.get());
				next_depth_nodes.emplace(curr->right.get());	
			}	
		}
		
		if( !this_level.empty()) {
			result.emplace_back(this_level);
		}
		curr_depth_nodes = next_depth_nodes;
	}
	return result;
}
// Time complexity: O(n). Space complexity: O(m), where m is maximum number of nodes at any single depth.


9.7 Compute Binary Tree Nodes in order of increasing depth

Given a binary tree, return an array consisting of the keys
at the same level. Keys should appear in the order of the
corresponding node's depths, breaking ties from left to right.

vector<vector<int>> BinaryTreeDepthOrder( 
	const unique_ptr<BinaryTreeNode<int>> & tree ) {
	
	queue<BinaryTreeNode<int>*> curr_depth_nodes({tree.get()});
	vector<vector<int>> result;
	
	while ( !curr_depth_nodes.empty() ) {
		queue<BinaryTreeNode<int>*> next_depth_nodes;
		vector<int> this_level;
		while ( !curr_depth_nodes.empty() ) {
			auto curr = curr_depth_nodes.front();
			curr_depth_nodes.pop();
			if (curr) {
				this_level.emplace_back(curr->data);
				// defer the null checks to the null test above
				next_depth_nodes.emplace(curr->left.get());
				next_depth_nodes.emplace(curr->right.get());
			}
		}
		if ( !this_level.empty() ) {
			result.emplace_back(this_level);
		}
		curr_depth_node = next_depth_nodes;
	}
	return result;
}
// Time complexity O(n)  Space: O(m), m is max nodes at any single depth





9.9 Implement a queue using stacks

Queue insertion and deletion follows first-in, first-out semantics;
stack insertion and deletion is last-in, first-out.
How would you implement a queue given a library implementing stacks?

class Queue {
 public:
	void Enqueue(int x) { enq_.emplace(x); }
	
	int Dequeue() {
		if (deq_.empty()) {
			while( !enq.empty()) {
				deq_.emplace(enq_.top());
				enq_.pop();
			}
		}
	}
	
	if (deq_.empty()) {
		throw length_error("empty queue");
	}
	int ret = deq_.top();
	deq_.pop();
	return ret;
	
 private:
	stack<int> enq_, deq_;
	
};
// Time: O(m) for m operations






///////////////////////////////////////////////////////
Chapter 10: Binary Trees
///////////////////////////////////////////////////////

// Binary trees boot camp

void TreeTraversal( const unique_ptr<BinaryTreeNode<int>> & root) {
	
	if (root) {
		
		// Preorder: processes root before traversal 
		// left and right of children
		cout << "Preorder: " << root->data << endl;		
		TreeTraversal(root->left);
		
		// Inorder: processes root after traversal of
		// left child and before right child traversal
		cout << "Inorder: " << root->data << endl;
		TreeTraversal(root->right);
		
		// Postorder: processes the root after of left
		// and right children
		cout << "Postorder: " << root->data << end;		
	}
}

// Time complexity of each approach is O(n) where n is
// number of nodes in the tree


10.1 Test if a binary tree is height-balanced

A binary tree is said to be height-balanced if for each
node in the tree, the difference in the height of its
left and right subtrees is at most one

Write a program that takes as input the root of a
binary tree and checks whether the tree is 
height-balanced

struct BalancedStatusWithHeight {
	bool balanced;
	int height;
};

bool IsBalanced( const unique_ptr<BinaryTreeNode<int>> &tree) {
	return CheckBalanced(tree).balanced;
}

BalancedStatusWithHeight CheckBalanced(
	const unique_ptr<BinaryTreeNode<int>> &tree) {
	
	if (tree == nullptr) {
		return {true, -1}; // base case
	}
	
	auto left_result = CheckBalanced(tree->left);
	if (!left_result.balanced) {
		return {false, 0}; // left subtree not balanced
	}
	
	auto right_result = CheckBalanced(tree->right);
	if (!right_result.balanced) {
		return {false, 0}; //right subtree not balanced
	}
	
	bool is_balanced = abs(left_result.height - right_result.height) <= 1;
	int height = max(left_result.height, right_result.height) + 1;
	return {is_balanced, height};	
}	
// Time complexity: O(n)


10.2 Test if a binary tree is symmetric 

Write a program that checks whether a binary tree is symmetric

bool IsSymmetric(const unique_ptr<BinaryTreeNode<int>>& tree) {
	return tree == nullptr || CheckSymmetric(tree->left, tree->right);
}

bool CheckSymmetric( const unique_ptr<BinaryTreeNode<int>> & subtree_0,
					 const unique_ptr<BinaryTreeNode<int>> & subtree_1 ) {
	if ( subtree_0 == nullptr && subtree_1 == nullptr ) {
		return true;
	}
	else if ( subtree_0 != nullptr && subtree_1 != nullptr ) {
		return subtree_0->data == subtree_1->data &&
			   CheckSymmetric(subtree_0->left, subtree_1->right) &&
			   CheckSymmetric(subtree_0->right, subtree_1->left);
	}
	// one subtree is empty, and the other is not
	return false;
}
// Time: O(n)   Space: O(h)
// n is number of nodes in tree and h is height of tree

	
	
10.3 Compute lca lowest common ancestor (LCA) in a
     Binary Tree

Design an algorithm for computing the LCA of two nodes in 
a binary tree in which nodes do not have a parent field

// EPI method 1

struct Status {
	int num_target_nodes;
	BinaryTreeNode<int> * ancestor;
};

BinaryTreeNode<int> * LCA( const unique_ptr<BinaryTreeNode<int>> & tree,
						  const unique_ptr<BinaryTreeNode<int>> & node0,
						  const unique_ptr<BinaryTreeNode<int>> & node1 ) {
	
	return LCAHelper(tree, node0, node1).ancestor;						  
}

Status LCAHelper( const unique_ptr<BinaryTreeNode<int>> & tree,
				  const unique_ptr<BinaryTreeNode<int>> & node0,
				  const unique_ptr<BinaryTreeNode<int>> & node1 ) {
	// base case
	if ( tree == nullptr ) {
		return {0, nullptr};
	}
	
	auto left_result = LCAHelper( tree->left, node0, node1 );
	if (left_result.num_target_nodes == 2) {
		return left_result; // found both nodes in left subtree
	}
	
	auto right_result = LCAHelpter( tree->right, node0, node1 );
	if (right_result.num_target_nodes == 2) {
		return right_result;
	}
	int num_target_nodes = left_result.num_target_nodes + 
						   right_result.num_target_nodes + 
						   (tree == node0) + (tree == node1);
	return {num_target_nodes, num_target_nodes == 2 ? tree.get() : nullptr};					   	
}
// Time: O(n)  Space: O(h)


// Leetcode method
Given the following binary search tree:  
root = [3,5,1,6,2,0,8,null,null,7,4]

        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4

Example 1:
Input: root, p = 5, q = 1
Output: 3
Explanation: The LCA of of nodes 5 and 1 is 3.

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    
    if (!root || !p || !q) {
        return NULL;
    }
    
    if (root == p || root == q) {
        return root;
    }
    
    TreeNode* l = lowestCommonAncestor(root->left, p, q);
    TreeNode* r = lowestCommonAncestor(root->right, p, q);
    
    if (l && r) {
        return root;
    }
    
    return l ? l : r;
}



10.4 Compute the LCA when nodes have parent pointers

Given two nodes in a binary tree, design an algorithm that
computes their LCA. Assume that each node has a parent pointer.

BinaryTreeNode<int> * LCA(const unique_ptr<BinaryTreeNode<int>> & node_0,
						  const unique_ptr<BinaryTreeNode<int>> & node_1) {
	
	auto *iter_0 = node_0.get(), *iter_1 = node_1.get();
	int depth_0 = GetDepth(iter_0), depth_1 = GetDepth(iter_1);
	// Makes iter_0 as the deeper node in order to simplify the code
	if (depth_1 > depth_0) {
		swap(iter_0, iter_1);
	}
	// Ascends from teh deeper node
	int depth_diff = abs(depth_0 - depth_1);
	while ( depth_diff-- ) {
		iter_0 = iter_0->parent;
	}
	// Ascends both nodes until LCA 
	while ( iter_0 != iter_1 ) {
		iter_0 = iter_0->parent, iter_1 = iter_1->parent;
	}
	return iter_0;
}

int GetDepth( const BinaryTreeNode<int> * node ) {
	int depth = 0;
	while ( node->parent ) {
		++depth, node = node->parent;
	}
	return depth;	
}
// Time: O(h)  Space: O(1)



10.5 Sum the root to leaf paths in a binary tree

Design an algorithm to compute the sum of the binary numbers
represented by the root to leaf paths.

int SumRootToLeaf( const unique_ptr<BinaryTreeNode<int>> & tree ) {
	
	return SumRootToLeafHelper( tree, 0 );
}

int SumRootToLeafHelper( const unique_ptr<BinaryTreeNode<int>> & tree,
						 int partial_path_sum ) {
	// base case
	if (tree == nullptr) {
		return 0;
	}
	
	partial_path_sum = partial_path_sum * 2 + tree->data;
	if (tree->left == nullptr && tree->right == nullptr) { // leaf
		return partial_path_sum;
	}
	// Non-leaf
	return SumRootToLeafHelper(tree->left, partial_path_sum) +
		   SumRootToLeafHelper(tree-right, partial_path_sum);	
}
// Time: O(n)  Space: O(h)



10.6 Find a root to leaf path with specified sum

Write a program which takes as input an integer and a binary 
tree with integer node weights, and checks if there exists a
leaf whose path weight equals the given integer.

bool HasPathSum( const unique_ptr<BinaryTreeNode<int>> & tree, 
				 int target_sum ) {
	
	return HasPathSumHelper(tree, 0, target_sum);
}

bool HasPathSumHelper( const unique_ptr<BinaryTreeNode<int>> & node,
					   int partial_path_sum, int target_sum ) {
	// base case
	if (node == nullptr) {
		return false;
	}
	partial_path_sum += node->data;
	if (node->left == nullptr && node->right == nullptr) { // leaf
		return partial_path_sum == target_sum; 
	}
	// Non-leaf
	return HasPathSumHelper(node->left, partial_path_sum, target_sum) ||
		   HashPathSumHelper(node->right, partial_path_sum, target_sum);
}
// time complexity: O(n)  Space: O(h)



10.7 Implement an inorder traversal without recursion

Write a program which takes as input a binary tree and performs an
inorder traversal of the tree. Do not use recursion. Nodes do not
contain parent references.

vector<int> BSTInSortedOrder( const unique_ptr<BSTNode<int>> & tree ) {
	// declare variables
	stack<const BSTNode<int> *> s;
	const auto* curr = tree.get();
	vector<int> result;
	
	while (!s.empty() || curr) {
		if (curr) {
			s.push(curr);
			// Going left
			curr = curr->left.get();
		}
		else {
			// Going up
			curr = s.top();
			s.pop();
			result.emplace_back(curr->data);
			// Going right
			curr = curr->right.get();
		}
	}
	return result;
}
// Time: O(n)  Space: O(h)


10.8 Implement a preorder traversal without recursion

Write a program which takes as input a binary tree and
performs a preorder traversal of the tree. Do not use
recursion. Nodes do not contain parent references.

vector<int> PreorderTraversal( 
				const unique_ptr<BinaryTreeNode<int>> & tree ) {
	// Declare variables
	stack<BinaryTreeNode<int>*> path;
	path.emplace(tree.get());
	vector<int> result;
	
	while ( !path.empty() ) {
		auto curr = path.top();
		path.pop();
		if ( curr != nullptr ) {
			result.emplace_back(curr->data);
			path.emplace(curr->right.get());
			path.emplace(curr->left.get());
		}
	}
	return result;	
}
// Time: O(n)  Space: O(h) 



10.9 Compute the kth node in an inorder traversal
// hint: use the divide and conquer principle
Write a program that efficiently computes the kth node
appearing in an inorder traversal. Assume that each node
stores the number of nodes in the subtree rooted at that
node.

const BinaryTreeNode<int> * FindKthNodeBinaryTree( 
		const unique_ptr<BinaryTreeNode<int>> &tree, int k ) {
	
	const auto * iter = tree.get();
	while (iter != nullptr) {
		int left_size = iter->left ? iter->left->size : 0;
		if (left_size + 1 < k) { //kth node in the right subtree of iter
			k -= (left_size + 1);
			iter = iter->right.get();
		}
		else if (left_size == k - 1) { // kth is iter itself
			return iter;
		}
		else { // kth node must in the left subtree
			iter = iter->left.get();
		}
	}
	// if k is between 1 and the tree size, this line is unreachable
	return nullptr;
}
// Time: O(h) since we descend the tree in each iteration


10.10 Compute the successor

Design an algorithm that computes the successor of a node
in a binary tree. Assume that each node stores its parent.

BinaryTreeNode<int> * FindSuccessor( 
	const unique_ptr<BinaryTreeNode<int>> & node) {
	
	auto * iter = node.get();
	if ( iter->right != nullptr ) {
		// successor is the leftmost element in node's right subtree
		iter = iter->right.get();
		while ( iter->left ) {
			iter = iter->left.get();
		}
		return iter;
	}
	
	while (iter->parent != nullptr && iter->parent->right.get() == iter) {
		
		iter = iter->parent;
	}
	// A return value of nullptr means node does not have successor
	// it is rightmost node in the tree
	return iter->parent;
}
// Time: O(h), where h is the height of the tree






///////////////////////////////////////////////////////
Chapter 11: Heap
///////////////////////////////////////////////////////

Heap (aka a priority queue) is a specialized binary
tree (complete binary tree). Max-heap supports: 
O(logn)insertions, 
O(1) time lookup for max element, and 
O(log n) deletion of the max element.
O(n) time complexity for searching 

// Use a heap when we care about the largest or 
// smallest elements and don't need fast lookup,
// delete, or search operations for arbitrary elements


// Heap methods:
top();
pop();
priority_queue // heap implementation in C++

// Compute k largest? Use min-heap
// Compute k smallest? Use max-heap


Heap boot camp problem:

Write a program which takes a sequence of strings presented
in "streaming" fashion: you cannot back up to read an earlier 
value. Your program must compute the k longest strings in the
sequence. All that is required is the k longest strings -- it 
is not required to order these strings.

vector<string> TopK (int k, isstringstream * stream) {
	// declaration
	priority_queue<string, vector<string>, function<bool(string, string)>>
	min_heap(
	[](const string &a, const string &b) {return a.size() >= b.size();});

	string next_string;
	while (*stream >> next_string) {
		min_heap.emplace(next_string);
		if (min_heap.size() > k) { // order strings by length
			min_heap.pop();  //remove shortest string
		}
	}
	vector<string> result;
	while (!min_heap.empty()) {
		result.emplace_back(min_heap.top());
		min_heap.pop();
	}
	return result;
}
// Time: O(nlogk) 



11.1 Merge Sorted Files

Write a program that takes as input a set of sorted sequences and
computes the union of these sequences as a sorted sequence. For
example, if the input is <3,5,7>,<0,6>, and <0,6,28>, then the 
output is <0,0,3,5,6,6,7,28>.

struct IteratorCurrentAndEnd {
	bool operator>( const IteratorCurrentAndEnd & that ) const {
		return *current > *that.current;
	}
	
	vector<int>::const_iterator current;
	vector<int>::const_iterator end;
};	

vector<int> MergeSortedArrays(
				const vector<vector<int> & sorted_arrays) {
		// Create a min_heap
		priority_queue<IteratorCurrentAndEnd, 
			vector<IteratorCurrentAndEnd>, greater<>> min_heap;
		
		for (const vector<int> & sorted_array : sorted_array) {
			if (!sorted_array.empty()) {
				min_heap.emplace( IteratorCurrentAndEnd{sorted_array.cbegin(), sorted_array.cend()});
			}
		}
		
		vector<int> result;
		while( !min_heap.empty() ) {
			auto smallest_array = min_heap.top();
			min_heap.pop();
			if (smallest_array.current != smallest_array.end) {
				result.emplace_back(*smallest_array.current);
				min_heap.emplace(IteratorCurrentAndEnd{
					next(smallest_array.current), smallest_array.end});
			}
		}
	return result;
}
// k is the number of input sequence
// Time: O(n log k)  Space: O(k)







11.2 Sort an increasing-decreasing array

Design an efficient algorithm for sorting a 
k-increasing-decreasing array.

vector<int> SortKIDArray( const vector<int> &A) {
	// Decomposes A into a set of sorted arrays
	vector<vector<int>> sorted_subarrays;
	typedef enum { INCREASING, DECREASING } SubarrayType;
	SubarrayType subarray_type = INCREASING;
	int start_idx = 0;
	for ( int i = 1; i <= A.size(); ++i ) {
		if (i == A.size() || 
			(A[i - 1] < A[i] && subarray_type == DECREASING) ||
			(A[i - 1] >= A[i] && subarray_type == INCREASING)) {
			if (subarray_type == INCREASING) {
				sorted_subarrays.emplace_back(A.cbegin() + start_idx, Acbegin() + i);
			}	
			else {
				sorted_arrays.emplace_back(A.crbegin() + A.size() - i,
										   A.crbegin() + A.size() - start_idx);
			}
			start_idx = i;
			subarray_type = (subarray_type == INCREASING ? DECREASING : INCREASING);
		}
    }
	
	return MergeSortedArrays(sorted_subarrays);
}

// Time complexity: O(nlogk) time



11.3 Sort an almost-sorted array

Write a program which takes as input a very long sequence of
numbers and prints the numbers in sorted order. Each number is
at most k away from its correctly sorted position.

void SortedAlmostSortedData( istringstream* sequence, int k ) {
	// Create priority queue for min_heap
	priority_queue<int, vector<int>, greater<>> min_heap;
	
	int x;
	for (int i = 0; i < k && *sequence >> x; ++i) {
		min_heap.push(x);
	}
	// For every new element, add it to min_heap, extract smallest
	while( *sequence >> x ) {
		min_heap.push(x);
		cout << min_heap.top() << endl;
		min_heap.pop();
	}
	// sequence is exhausted, iteratively extracts remaining elements
	while (!min_heap.empty()) {
		cout << min_heap.top() << endl;
		min_heap.pop();
	}
}
// Time: O(n log k )  Space: O(k)

11.4 Compute the k closest stars
// RAM limitation

Compute the k stars which are closest to Earth.

struct Star {
	bool operator<( const Star & that ) const {
		return Distance() < that.Distance();
	}
	
	double Distance() const { return sqrt(x * x + y * y + z * z); }
	
	double x, y, z;
};

vector<Star> FindClosestKStars( int k, istringstream* stars ) {
	// max_heap to store closest k stars seen so farcalloc
	priority_queue<Star, vector<Star>> max_heap;
	
	string line;
	while( getline( *stars, line ) ) {
		stringstream line_stream(line);
		array<double, 3> data; // stores x, y, and z.
		for (int i = 0; i < 3; ++i) {
			string buf;
			getline(line_stream, buf, ',');
			data[i] = stod(buf);
		}
		// Add each star to the max-heap. If max-heap size
		// exceeds k, remove the maximum element from the
		// max-heap
		max_heap.emplace(Star{data[0], data[1], data[2]});
		if (max_heap.size() == k + 1) {
			max_heap.pop();
		}
	}
		// Iteratively extract frm max-heap, which yields 
		// the stars sorted from furthest to closest
	vector<Star> closest_stars;
	while (!max_heap.empty()) {
		closest_stars.emplace_back(max_heap.top());
		max_heap.pop();
	}
	return {closest_stars.rbegin(), closest_stars.rend()};
}
// Time: O(n log k)  Space: O(k)


11.7 Implement a stack API using a heap

class Stack {
 public:
	void Push(int x) {max_heap_.emplace(ValueWithRank{x, timespace_++}); }
	
	int pop() {
		if (max_heap_.empty()) {
			throw length_error("empty stack");
		}
		int val = max_heap_.top().value;
		max_heap_.pop();
		return val;
	}
	
	int Peek() const { return max_heap_.top().value; }
	
 private:
	int timestamp_ = 0;
	
	struct ValueWithRank {
		int value, rank;
		
		bool operator<(const ValueWithRank & that) const {
			return rank < that.rank;
		}
	};
	priority_queue<ValueWithRank, vector<ValueWithRank>> max_heap_;
};
// O(log n) for push, pop, extract-max from max-heap



///////////////////////////////////////////////////////
Chapter 12: Searching
///////////////////////////////////////////////////////

Algorithms that are used to solve Google problems 
include sorting (plus searching and binary
search), divide-and-conquer, dynamic programming, 
memoization, greediness, recursion or
algorithms linked to a specific data structure.

Search boot camp:

Perform search on student GPA

struct Student 
{
	string name;
	double grad_point_average;
};

const static function<bool(const Student&, const Student&)> ComGPA = [](
	const Student& a, const Student& b) 
{	
	if (a.grade_point_average != b.grad_point_average) {
		return a.grade_point_average > b.grad_point_average;
	}
	return a.name < b.name;
}

bool SearchStudent(
	const vector<Student>& students, const Student& target,
	const function<bool(const Student&, const Student&)>& comp_GPA) 
{
   return binary_search(students.begin(), students.end(), target, comp_GPA);

}





12.1 Search a sorted array for first occurence of k

Write a method that takes a sorted array and a key and
returns the index of the first occurence of that key

int SearchFirstOfK( const vector<int> &A, int k ) {
	// Instantiate variables
	int left = 0;
	int right = A.size() - 1;
    int result = -1;
	// [left : right] is the candidate set
	while (left <= right) {
		int mid = left + ((right - left) / 2);
		if (A[mid] > k) {
			right = mid - 1;
		}
		else if (A[mid] == k) {
			result = mid;
			// nothing to right of mid can be first occurence of k
			right = mid - 1;
		}
		else { // A[mid] < k
			left = mid + 1;
		}
	}
	return result;
}
// Time complexity: O(log n) because each iteration
// reduces the size of candidate set by half




12.2 Search a sorted array for entry equal to its index

Design an efficient algorithm that takes a sorted
array of distinct intergers and returns an index i
such that the element at index i equals i. For example,
when the input is <-2,0,2,3,6,7,9> your algorithm
should return 2 or 3.

int SearchEntryEqualToIndex( const vector<int> &A) {
	// Instantiate variables
	int left = 0;
	int right = A.size() - 1;
	
	while (left <= right) {
		int mid = left + ((right - left) / 2);
		int difference = A[mid] - mid;
		// A[mid] == mid if and only if difference == 0
		if (difference == 0) {
			return mid;
		}
		else if ( difference > 0 ) {
			right = mid - 1;
		}
		else {
			left = mid + 1;
		}	
	}
	return -1;
}
// O(log n) time complexity



12.3 Search a cyclically sorted array

// Hint: Use the divide and conquer principle

Design an O(log n) algorithm for finding the position
of the smallest element in a cyclically sorted array

int SearchSmallest( const vector<int> &A ) {
	
	int left = 0;
	int right = A.size() - 1;
	
	while (left < right) {
		int mid = left + ((right - left) / 2);
		if (A[mid] > A[right]) {
			left = mid + 1;
		}
		else if (A[mid] < A[right]) {
			right = mid;
		}
	}
	// loop ends when left == right
	return left;
}
// Time complexity: O(log n)



// Generalized search that don't use binary search principle

12.6 Search in a 2D sorted array

Design an algorithm that takes a 2D sorted array and
a number and checks whether that number appears in
the array.

bool MatrixSearch( const vector<vector<int>> &A, int x ) {
	
	// Create and initialize variables
	int row = 0;
	int col = A[0].size() - 1;
	
	// while row is less than A's 2D array size
	// and while column is greater or equal to 0
	while (row < A.size() && col >= 0) {
		// if x is found return true!
		if (A[row][col] == x) {
			return true;
		}
		else if (A[row][col] < x) {
			++row; // increment/eliminate row
		}
		else { //if (A[row][col] > x) 
			--col;
		}
	}
	return false;
}
// Time complexity is O(m + n) 

12.7 Find the min and max elements simultaneously

Given an array of comparable objects, you can find the 
min or the max of the elements in the array with n - 1
comparisons, where n is the length of the array.
For example, if A = <3, 2, 5, 1, 2, 4>, you should return
1 for the min and 5 for the max.

struct MinMax {
	int min, max;
};

MinMax FindMinMax( const vector<int> & A ) {
	if ( A.size() <= 1 ) {
		return { A.front(), A.fron() };
	}
	
	pair<int, int> global_min_max = minmax( A[0], A[1] );
	// process two elements at a time.
	for ( int i = 2; i + 1 < A.size(); i += 2 ) {
		pair<int, int> local_min_max = minmax(A[i], A[i + 1];
		global_min_max = {min(global_min_max.first, local_min_max.first),
						  max(global_min_max/second, local_min_max.second)};
	}
	// if there is odd number of elements in the array, we still
	// need to compare the last element with the exist answer
	if (A.size() % 2) {
		global_min_max = {min(global_min_max.first, A.back()),
					      max(global_min_max.second, A.back())};
	}
	return {global_min_max.first, global_min_max.second};	
}
// Time complexity is O(n). Space complexity is O(1)





/////////////////////////////////////////////////////
Chapter 13: Hash Tables
/////////////////////////////////////////////////////


Optimally, O(1) or constant time on average for
lookups, insertions. 
Deletions have O(1 + n/m) time, where n is number of
objects and m is the length of the array.

// Hash table boot camp

Write a program that takes as input a set of words
and returns groups of anagrams for those words. Each
group must contain at least two words.
For example, debitcard and badcredit are anagrams

vector<vector<string>> FindAnagrams(const vector<string> & dictionary) {
	
	unordered_map<string, vector<string>> sorted_string_to_anagrams;
	
	for (const string& s : dictionary) {
		//sort string, uses it as a key, and then
		// appends the original string as another
		// value into hash table
		string sorted_str(s);
		sort(sorted_str.begin(), sorted_str.end());
		sorted_string_to_anagrams[sorted_str].emplace_back(s);		
	}
	
	vector<vector<string>> anagram_groups;
	for (const auto& p : sorted_string_to_anagrams) {
		if (p.second.size() >= 2) { // Found anagrams
		anagram_groups.emplace_back(p.second);
	}
	return anagram_groups;
	
}
// Time complexity is O(nm log m)


13.1 Test For Palindromic Permutations

Write a program to test whether letters forming a string can be
permuted to form a palindrome. For example, "edified" can be 
permuted to form "deified".

bool CanFormPalindrome( const string & s ) {
	// create a hash map
	unordered_map<char, int> char_frequencies;
	// compute the frequency of each char in s
	for (char c : s) {
		++char_frequencies[c];
	}
	// a string can be permuted as palindrome if and only if the
	// number of chars whose frequencies is odd is at most 1
	int odd_frequency_count = 0;
	return none_of(begin(char_frequencies), end(char_frequencies),
				  [&odd_frequency_count](const auto & p) {
		
		return (p.second % 2) && ++odd_frequency_count > 1;		
	});	
}
// Time: O(n), where n is the length of the string
// Space: O(c), where c is the number of distinct characters


13.2 Is an anonymous letter constructible?

Write a program which takes text for an anonymous letter and text
for a magazine and determines if it is possible to write the 
anonymous letter using the magazine.

bool IsLetterConstructibleFromMagazine( const string & letter_text,
										const string & magazine_text ) {
	unordered_map<char, int> frequency_map;
	// Compute the frequencies for all chars in letter_text
	for (char c : letter_text) {
		++frequency_map[c];
	}
	// Check if the characters in magazine_text can cover characters
	// in frequency_map
	for (char c : magazine_text) {
		auto it = frequency_map.find(c);
		if (it != frequency_map.cend()) {
			--it->second;
			if (it->second == 0) {
				frequency_map.erase(it);
				if (frequency_map.empty()) {
					break; // all characters for letter_text are match
				}				
			}
		}		
	}
	// empty frequency_map for letters means every char in letter_text
	// can be covered by a character in magazine_text
	return frequency_map.empty();
}
// Time: O(m + n), where m and n are number of characters in letter
// and magazine respectively.
// Space: O(L), where L is the number of distinct characters appearing 
// in letter. It is the size of the hash table.


13.4 Compute LCA, optimizing for close ancestor

Design an algorithm for computing the LCA of two nodes in a
binary tree. The algorithm's time complexity should depend only
on the distance from the nodes to the LCA.

BinaryTreeNode<int> * LCA( const unique_ptr<BinaryTreeNode<int>> & node_0,
						   const unique_ptr<BinaryTreeNode<int>> & node_1 ) {
	// Declarations
	auto *iter_0 = node_0.get(), *iter_1 = node_1.get();
	unordered_set<const BinaryTreeNode<int>*> nodes_on_path_to_root;
	// iterate
	while (iter_0 || iter_1) {
		// Ascend tree in tandem for these two nodes
		if (iter_0) {
			if (nodes_on_path_to_root.emplace(iter_0).second == false) {
				return iter_0;
			}
			iter_0 = iter_0->parent;
		}
		if (iter_1) {
			if (nodes_on_path_to_root.emplace(iter_1).second == false) {
				return iter_1;
			}
			iter_1 = iter_1->parent;
			iter_1 = iter_1->parent;
		}
	}
	throw invalid_argument("node_0 and node_1 are not in the same tree");						   
}
// Time O(D0 + D1) Space and Time where D0 is the distance from the LCA
// to the first node and D1 is the distance from the LCA to the second node
// In the worst-case, teh nodes are leaves whose LCA is the root, and the
// time and space is O(h)




13.6 Find the nearest repeated entries in an array

Write a program which takes as input an array and finds the distance
between a closest pair of equal entries. 

int FindNearestRepetition( const vector<string> & paragraph ) {
	// declare variables
	unordered_map<string, int> word_to_latest_index;
	int nearest_repeated_distance = numeric_limits<int>::max();
	
	for ( int i = 0; i < paragraph.size(); ++i ) {
		auto latest_equal_word = word_to_latest_index.find(paragraph[i]);
		if (latest_equal_word != word_to_latest_index.end()) {
			nearest_repeated_distance = 
				min(nearest_repeated_distance, i - latest_equal_word->second);
		}
		word_to_latest_index[paragraph[i]] = i;
	}
	return nearest_repeated_distance;	
}
// Time: O(n)  Space: O(d), d is number of distinct entries in array



13.7 Find the smallest subarray covering all values

Write a program which takes an array of strings and a set of strings,
and return the indices of the starting and ending index of a shortest
subarray of the given array that "covers" the set, i.e., contains all
strings in the set.

struct Subarray {
	int start, end;
};

Subarray FindSmallestSubarrayCoveringSet(
	const vector<string>  & paragraph, 
	const unordered_set<string> & keywords ) 
{
	// Create a hash table
	unordered_map<string, int> keywords_to_cover;
	
	for ( const string & keyword : keywords ) {
		++keywords_to_cover[keyword];
	}
	
	Subarray result = Subarray{-1, -1}; // result to store start/end indices
	int remaining_to_cover = keywords.size();
	
	for (int left = 0, right = 0; right < paragraph.size(); ++right) {
		if ( keywords.count(paragraph[right]) &&
				--keywords_to_cover[paragraph[right]] >= 0 ) {
			--remaining_to_cover;	
		}
		// Keeps advancing left until keywords_to_cover does not contain all
		// keywords
		while ( remaining_to_cover == 0 ) {
			if ( (result.start == -1 && result.end == -1) || 
				 right - left < result.end - result.start ) {
				result = { left, right };	 
			}
			if ( keywords.count(paragraph[left]) &&
				 ++keywords_to_cover[paragraph[left]] > 0 ) {
				++remaining_to_cover;	 
			}
			++left;
		}	
	}
	return result;
}
// Time: O(n), where n is the length of the array
	

	


13.9 Find the longest subarray with distinct entries

Write a program that takes an array and returns the
length of a longest subarray with the property that all
its elements are distinct. For example, if the array is
<f,s,f,e,t,w,e,n,w,e> then a longest subarray all whose
elements are distinct is <s,f,e,t,w>.

int LongestSubAWithDistEnt( const vector<int> &A) {
	// Records the most recent occurences of each entry
	unordered_map<int, size_t> most_recent_occurence;
	
	size_t longest_dup_free_start_idx = 0;
	size_t result = 0;
	
	for (size_t i = 0; i < A.size(); ++i) {
		auto dup_idx = most_recent_occurence.emplace(A[i], i);
		// Defer updating dup_idx until we see duplicates		
		if (!dup_idx.second) {
			if (dup_idx.first->second >= longest_dup_free_start_idx) {
				result = max(result, i - longest_dup_free_start_idx);
				longest_dup_free_start_idx = dup_idx.first->second + 1;
			}
		dup_idx.first->second = i;
		}	
	}
	result = max(result, A.size() - longest_dup_free_start_idx);
	return result;
}
// Time complexity is O(n), since we perform constant
// number of operations per element


13.10 Find the length of a longest contained interval

Write a program which takes as input a set of integers represented
by an array, and returns the size of a largest subset of integers 
in the array having the property that if two integers are in the 
subset, then so are all integers between them.

int LongestContainedRange( const vector<int> & A ) {
	// Declarations
	unordered_set<int> unprocessed_entries(A.begin(), A.end());
	
	int max_interval_size = 0;
	while ( !unprocessed_entries.empty() ) {
		int a = *unprocessed_entries.begin();
		unprocessed_entries.erase(a);
		// Finds lower bound of the largest range containing a
		int lower_bound = a - 1;
		
		while (unprocessed_entries.count(lower_bound)) {
			unprocessed_entries.erase(lower_bound);
			--lower_bound;
		}
		// Finds the upper bound of largest range containing a
		int upper_bound = a + 1;
		while (unprocessed_entries.count(upper_bound)) {
			unprocessed_entries.erase(upper_bound);
			++upper_bound;
		}
		max_interval_size = max(max_interval_size, upper_bound - lower_bound - 1);
	}
	return max_interval_size;
}
// Time: O(n), 


13.12 Compute all string decompositions

Write a program which takes as input a string(the "sentence") and
an array of strings (the "words"), and returns the starting indices
of substrings of the sentence string which are the concatenation of
all the strings in the words array. 

vector<int> FindAllSubstrings( const string& s, const vector<string>& words ) {
	// Declaration
	unordered_map<string, int> word_to_freq;
	// Iterate
	for (const string& word : words) {
		++word_to_freq[word];
	}
	
	int unit_size = words.front().size();
	vector<int> result;
	
	for (int i = 0; i + unit_size * words.size() <= s.size(); ++i) {
		if (MatchAllWordsInDict(s, word_to_freq, i, words.size(), unit_size)) {
			result.emplace_back(i);
		}
	}
	return result;	
}

bool MatchAllWordsInDict( const string& s, 
						  const unordered_map<string, int>& word_to_freq,
						  int start, int num_words, int unit_size ) {
	// Create a hash table
	unordered_map<string, int> curr_string_to_freq;						  
	for (int i = 0; i < num_words; ++i) {
		string curr_word = s.substr(start + i * unit_size, unit_size);
		auto iter = word_to_freq.find(curr_word);
		if (iter == word_to_freq.end()) {
			return false;
		}
		++curr_string_to_freq[curr_word];
		if (curr_string_to_freq[curr_word] > iter->second) {
			// curr_word occurs too many times for a match to be possible
			return false;
		}
	}
	return true;
}
// Time O(Nnm)



/////////////////////////////////////////////////////
Chapter 14: Sorting
/////////////////////////////////////////////////////


sort(); 
//sort() uses O(n log n) time complexity
// variant of quicksort, O(log n) space 

Sort an array of students by GPA. Using compare function

struct Student {
	bool operator <(const Student &that) const { return name < that.name;)
	
	string name;
	double grade_point_average;	
};

void SortByGPA(vector<Student> * students) {
	sort(students->begin(), students->end(),
		[](const Student &a, const Student &b) {
			return a.grade_point_average >= b.grade_point_average;
		});
}

void SortByName(vector<Student> * students) {
	// Rely on the operator < defined in student.
	sort(students->begin(), students->end());
}

// Time complexity: O(n log n)
// Space complexity O(1) space






14.1 Compute the intersection of two sorted arrays

Write a program which takes as input two sorted arrays,
and returns a new array containing elements that are
present in both of input arrays. For example, the input
is <2,3,4,5,6,7,8,10> and <5,5,6,7,8,12,13>, then the
output is <5,6,7,8>


// Best method in O(n) or linear time 

vector<int> IntersectTwoSortedArrays(const vector<int> &A,
									 const vector<int> &B) {
	// Instantiate variables
	vector<int> intersection_A_B;
	int i = 0; // first pointer
	int j = 0; // second pointer
	
	while (i < A.size() && j < B.size()) {
		if (A[i] == B[j] && (i == 0 || A[i] != A[i - 1])) {
			intersection_A_B.emplace_back(A[i]);
			++i, ++j;
		}
		else if (A[i] < B[j]) {
			++i;
		}
		else { //A[i] > B[j]
			++j;
		}
	}
	return intersection_A_B;
}
// Time complexity: O(m + n) because we spend O(1)
// time per input array element


// Binary search method for arrays intersection problem
vector<int> IntersectTwoSortedArrays(const vector<int> &A,
									 const vector<int> &B) {
			
	vector<int> intersection_A_B;
	
	for (int = 0; i < A.size(); ++i) {
		if ((i == 0 || A[i] != A[i - 1]) &&
			 binary_search(B.cbegin(), B.cend(), A[i])) {
			intersection_A_B.emplace(A[i]);	 
		}
	}
	return intersection_A_B;
}

// Time complexity: O(m log n), where m is the length
// of the array being iterated over.


// Shorter variable naming way
vector<int> IntersectArrays(const vector<int> &A,
							const vector<int> &B) {
		
	vector<int> result;
	int i = 0, j = 0;
	
	while (i < A.size() && j < B.size()) {
		if (A[i] == B[j] && (i == 0 || A[i] != A[i - 1])) {
			result.emplace_back(A[i]);
			++i, ++j;
		}
		else if (A[i] < B[j]) {
			++i;
		}
		else { // A[i] > B[j]
			++j;
		}
	}	
	return result;	
}
// Time complexity: O(m + n). 
// We spend O(1) time per input array element.


// Brute force approach

vector<int> IntersectTwoSortedArrays(const vector<int> &A,
									const vector<int> &B) {
	// Instantiate vector of type int
	vector<int> intersection_A_B;
	
	for (int i = 0; i < A.size(); ++i) {
		if (i == 0 || A[i] != A[i - 1]) {
			for (int b : B) {
				if (A[i] == b) {
					intersection_A_B.emplace_back(A[i]);
					break;
				}
			}
		}
	}
	return intersection_A_B;
}
// Time complexity: O(mn)





14.2 Merge Two sorted arrays

Write a program which takes as input two sorted arrays
of integers, and updates the first array to the combined
entries of the two arrays in sorted order.

void MergeTwoSortedArrays(int A[], int m, int B[], int n) {
	
	int a = m - 1; 
	int b = n - 1;
	int write_idx = m + n - 1;
	
	while (a >= 0 && b >= 0) {
		A[write_idx--] = A[a] > B[b] ? A[a--] : B[b--];
	}
	while (b >= 0) {
		A[write_idx--] = B[b--];
	}
}
// Time complexity is O(m + n). O(1) additional space.







14.3 Remove First Name Duplicates string

Design an efficient algorithm for removing all first
name duplicates from an array. For example, if the input
is <(Ian,Botham),(David,Gower),(Ian,Bell),(Ian,Chappell)>, 
one result could be <(Ian,Bell),(David,Gower)>;
<(David,Gower),(Ian,Botham)> would also be acceptable

struct Name {
	
	bool operator==(const Name &that) const {
		return first_name == that.first_name;
	}
	
	bool operator<(const Name &that) const {
		if (first_name != that.first_name) {
			return first_name < that.first_name;
		}
		return last_name < that.last_name;
	}
	
	string first_name, last_name;	
};

void EliminateDuplicate(vector<Name> *A) {
	
	sort(A->begin(), A->end());
	// unique() removes adjacent duplicates
	A->erase(unique(A->begin(), A->end(), A->end());
}
// Time complexity: O(n log n). Space complexity O(1)



14.4 Render a calendar

Write a program that takes a set of events, and 
determines the maximum number of events that take place
concurrently.

struct Event {
	int start, finish;
};

struct Endpoint {
	
	bool operator<(const Endpoint & e) const {
	// if times are equal, an endpoint that starts an
	// interval comes first
		return time != e.time ? time < e.time : (isStart && !e.isStart);
	}
	
	int time;
	bool isStart;
};

int FindMaxSimultaneousEvents(const vector<Event> &A) {
	// Builds an array of all endpoints
	vector<Endpoint> E;
	
	// for loop over endpoints and emplace_back
	// events from start to finish
	for (const Event& event : A) {
		E.emplace_back(Endpoint{event.start, true});
		E.emplace_back(Endpoint{event.finish, false});
	}
	// Sorts the endpoint array according to time
	sort(E.begin(), E.end());
	
	//Track the number of simultaneous events, and
	//record the maximum number of simultaneous events
	int max_num_simul_events = 0;
	int num_simul_events = 0;
	
	// for loop over endpoints 
	for (const Endpoint& endpoint : E) {
		// if event has started increment 
		// num_simul_events counter/index
		if (endpoint.isStart) {
			++num_simul_events;
			max_num_simul_events = 
				max(num_simul_events, max_num_simul_events);
		}
		else {
			--num_simul_events;
		}
	}
	return max_num_simul_events;	
}
// Sorting the endpoint array takes O(n log n) time;
// iterating through the sorted array takes O(n),
// yielding an O(n log n) time complexity
// Time complexity is O(n), which is the size of 
// the endpoint array


14.5 Merging or merge Interval or insert interval
Write a program which takes as input an array of disjoint
closed intervals with integer endpoints, sorted by increasing
order of left endpoint, and an interval to be added, and 
returns the union of the intervals in the array and the added
interval. Your result should be experssed as a union of disjoint
intervals sorted by left endpoint.

struct Interval {
	int left, right;
};

vector<Interval> AddInterval( const vector<Interval> & disjoint_intervals,
							  Interval new_interval) {
		size_t i = 0;
		vector<Interval> result;

	// Processe intervals in disjoint_intervals which come before new_interval
	while ( i < disjoint_intervals.size() && 
			new_interval.left > disjoint_intervals[i].right) {
				
		result.emplace_back(disjoint_intervals[i++];		
	}
	
	// Processes intervals in disjoint_intervals which overlap with	
	// new_interval.
	while (i < disjoint_intervals.size() &&
			new_interval.right >= disjoint_intervals[i].left) {
			
		new_interval = {min(new_interval.left, disjoint_intervals[i].left),
					max(new_interval.right, disjoint_intervals[i].right)};
		++i;
	}
	result.emplace_back(new_interval);
	
	// processes intervals in disjoint_intervals which come after
	// new_interval
	result.insert(result.end(), disjoint_intervals.begin() + i,
				  disjoint_intervals.end());
	
	return result;		
}
// Time complexity: O(n), since program spends constant time per entry
// Space: O(n)



14.9 Implement a fast sorting algorithm for lists( merge lists sorted )

Unlike arrays, lists can be merged in-place. Insertion in the middle 
of list is O(1)
// The following program implements a merge sort on 
// lists

Implement a routine algorithm which sorts lists
efficiently. It should be a stable sort, i.e., the
relative positions of equal elements must remain 
unchanged. 

shared_ptr<ListNode<int>> StableSortList(
						shared_ptr<ListNode<int>> L) {
	// Base cases: L is empty or a single node, nothing to do
	if (L == nullptr || L->next == nullptr) {
		return L;
	}	
	// find the midpoint of L using a slow and a fast
	// pointer
	share_ptr<ListNode<int>> pre_slow = nullptr;
	share_ptr<ListNode<int>> slow = L;
	share_ptr<ListNode<int>> fast = L;
	
	while (fast && fast->next) {
		pre_slow = slow;
		fast = fast->next->next;
		slow = slow->next;
	}
	
	pre_slow->next = nullptr; // Splits the list into two equal-sized lists
	
	return MergeTwoSortedLists(StableSortList(L), StableSortList(slow));
}
// Time complexity: O(n log n) same as mergesort.
// Space complexity: O(log n).

// StableSortList function for function call above
shared_ptr<ListNode<int>> MergeTwoSortedLists(shared_ptr<ListNode<int>> L1,
											  shared_ptr<ListNode<int>> L2) {
	// Creates a placeholder for the result
	shared_ptr<ListNode<int>> dummy_head(new ListNode<int>);
	auto tail = dummy_head;
	
	while (L1 && L2) {
		AppendNode(L1->data <= L2->data ? &L1 : &L2, &tail);	
	}
	// Appends the remaining nodes of L1 or L2
	tail->next = L1 ? L1 : L2;
	return dummy_head->next;	
}

void AppendNode( shared_ptr<ListNode<int>> *node,
			     shared_ptr<ListNode<int>> *tail) {
		
	(*tail)->next = *node; 
	*tail = *node;
	*node = (*node)->next;
}
// Time complexity is O(n + m). Space complexity: O(1)




///////////////////////////////////////////////////////
Chapter 15: Binary Search Trees (BST)
///////////////////////////////////////////////////////


Binary Search Trees have the special property that
the key stored at a node is greater than or equal to
the keys stored at the nodes of its left subtree and
less than or equal to the keys stored in the nodes 
of its right subtree. 

// Key lookup, insertion, and deletion take time
// proportional to the height of the tree, which can
// in worst-case be O(n), if insertions and deletions
// are naively implemented

BSTNode<int>* SearchBST(const unique_ptr<BSTNode<int>> &tree, int key) {
	// base case
	if (tree == nullptr) {
		return nullptr;
	}
	
	if (tree->data == key) {
		return tree.get();
	}
	return key < tree->data ? SearchBST(tree->left, key)
							 :SearchBST(tree->right, key);
}
// Time complexity is O(h), where h is the height of
// the tree

// Binary Search Tree methods
begin(); // traverse keys in ascending order
rbegin(); // traverse keys in descending order
*begin(); // yield smallest key in BST
*rbegin(); // yield largest key in BST
lower_bound(12)/upper_bound(3) // return the first
// element that is greater than or equal to/greater
// than the argument
equal_range(10); // return the range of values equal
				// to the argument
				

15.1 Test if a Binary Tree satisfies the BST property
validate binary search tree

Write a program that takes as input a binary tree and
checks if the tree satisfies the BST property.

// Recursive method 1 
bool IsBinaryTreeBST( const unique_ptr<BinaryTreeNode<int>> &tree ) {
	return AreKeysInRange(tree, numeric_limits<int>::min(),
						  numeric_limits<int>::max());
}

bool AreKeysInRange(const unique_ptr<BinaryTreeNode<int>> &tree,
					int low_range, int high_range) {
	if (tree == nullptr) {
		return true;
	}
	else if (tree->data < low_range || tree->data > high_range) {
		return false;
	}
	
	return AreKeysInRange(tree->left, low_range, tree->data) &&
		   AreKeysInRange(tree->right, tree->data, high_range);		
}
// Time complexity is O(n). Space complexity: O(h)

// Short method
class Solution {
public:
  bool isValidBST(TreeNode* root, long min = LONG_MIN, long max = LONG_MAX) {
    
	if (root == NULL) 
		return true;
	
    if (root->val <= min || root->val >= max) 
		return false;
	
    return isValidBST(root->left, min, root->val) && isValidBST(root->right, root->val, max);
  }
};
// Time complexity is O(n). Space complexity: O(h)

// Breadth First Search method 
struct QueueEntry {
	const unique_ptr<BinaryTreeNode<int>>& tree_node;
	int lower_bound, upper_bound;
};

bool IsBinaryTreeBST(const unique_ptr<BinaryTreeNode<int> &tree) {
	// Create a queue
	queue<QueueEntry> BFS_queue;
	BFS_queue.emplace(QueueEntry{tree, numeric_limits<int>::min(),
	                             numeric_limits<int>::max()});

								 
	while (!BFS_queue.empty()) {
		if (BFS_queue.front().tree_node.get()) {
			if (BFS_queue.front().tree_node->data < BFS_queue.front().lower_bound ||
				BFS_queue.front().tree_node->data > BFS_queue.front().upper_bound) {
			return false;
			}
			
			BFS_queue.emplace(QueueEntry{BFS_queue.front().tree_node->left,
										 BFS_queue.front().lower_bound,
										 BFS_queue.front().tree_node->data});
			
			BFS_queue.emplace(QueueEntry{BFS_queue.front().tree_node->right,
										 BFS_queue.front().tree_node->data,
										 BFS_queue.front().upper_bound});			
		}
		BFS_queue.pop();
	}
	return true;
}
// Time complexity is O(n). Space complexity: O(h)


15.2 Find the first key greater than a given value in a BST

Write a program that takes as input a BST and a value, and returns
the first key that would appear in an inorder traversal which is
greater than the input value. 

BSTNode<int> * FindFirstGreaterThanK ( const unique_ptr<BSTNode<int>> & tree,
									   int k ) {
	// Declarations
	BSTNode<int> *subtree = tree.get();
	BSTNode<int> *first_so_far = nullptr;
	while (subtree) {
		if (subtree->data > k ) {
			first_so_far = subtree->left.get();
		}
		else {
			subtree = subtree->right.get();
		}
	}
	return first_so_far;								   
}
// Time: O(h)  Space: O(1)




15.4 Compute the LCA in a BST

Design an algorithm that takes as input a BST and two
nodes, and returns the LCA of the two nodes.

BSTNode<int>* FindLCA(const unique_ptr<BSTNode<int> &tree,
					  const unique_ptr<BSTnode<int>> &s,
					  const unique_ptr<BSTNode<int>> &b) {
	
	auto *p = tree.get();
	while (p->data < s->data || p->data > b->data) {
		// keep searching since p is outside of [s,b]
		while (p->data < s->data) {
			p = p->right.get(); // LCA is in p's right child
		}
		
		while (p->data > b->data) {
			p = p->left.get(); // LCA is in p's left child
		}	
	}
	
	return p;
}
// Since we descend one level with each iteration, the
// time complexity is O(h), where h is the height of
// the tree.

// adding and removing entries from height-balanced
// BST on N nodes takes O(logN) time complexity



///////////////////////////////////////////////////////
Chapter 16: Recursion
///////////////////////////////////////////////////////


16.1 The Towers of Hanoi Problem

Write a program which prints a sequence of operations
that transfer n rings from one peg to another.

const int kNumPegs = 3;

void ComputeTowerHanoi(int num_rings) {
	
	array<stack<int>, kNumPegs> pegs;
	// Initialize pegs
	for (int i = num_rings; i >= 1; --i) {
		pegs[0].push(i);
	}
	
	computeTowerHanoiSteps(num_rings, pegs, 0, 1, 2);
	
	void ComputeTowerHanoiSteps(int num_rings_to_move,
								array<stack<int>, kNumPegs> &peg, int from_peg,
								int to_peg, int use_peg) {
		
		if (num_rings_to_move > 0) {
			ComputeTowerHanoiSteps(num_rings_to_move - 1, 
			                   pegs, from_peg, use_peg, to_peg);
			pegs[to_peg].push(pegs[from_peg].top());
			pegs[from_peg].pop();
			ComputeTowerHanoiSteps(num_rings_to_move - 1, pegs, use_peg, to_peg, from_peg);
		}		
	}	
}
// Time complexity is O(2^n)


16.3 Generate permutations

Write a program which takes as input an array of 
distinct integers and generates all permutations of
that array.

vector<vector<int> Permutations(vector<int> A) {
	
	vector<vector<int>> result;
	// Generate the first permutation in dictionary
	// order
	sort(A.begin(), A.end());
	
	do {
		
		result.emplace_back(A);
		
	} while (next_permutation(A.begin(), A.end()));

	return result;
}
// Time complexity is O(n x n!), since there are n!
// permutations


///////////////////////////////////////////////////////
Chapter 17: Dynamic Programming
///////////////////////////////////////////////////////


// DP is a general technique for solving optimization,
// search, and counting problems that can be 
// decomposed into subproblems

consider using DP whenever you have to make choices to 
arrive at the solution, specifically, when the solution
relates to subproblems


Like divide-and-conquer, DP solves the problem by
combining the solutions of multiple smaller problems,
but what makes DP different is that the same subproblem
may reoccur. A key to make DP efficient is caching
the results of intermediate computations.

int Fibonacci( int n ) {
	// base case
	if (n <= 1) {
		return n;
	}
	
	int f_minus 2 = 0, f_minus_1 = 1;
	
	for (int i = 2; i <= n; ++i) {
		int f = f_minus_2 + f_minus_1;
		f_minus_2 = f_minus_1;
		f_minus_1 = f;	
	}
	return f_minus_1;
}
// Time complexity: O(n). Space complexity: O(1).



Maximum subarray sum problem

int MaximumSubarray( const vector<int> & A ) {
	
	int min_sum = 0, max_sum = 0, running_sum = 0;
	
	for ( int i = 0; i < A.size(); ++i ) {
		running_sum += A[i]; // update running sum
		
		if ( running_sum < min_sum ) {
			min_sum = running_sum;
		}
		if (running_sum - min_sum > max_sum) {
			max_sum = running_sum - min_sum;
		}		
	}	
	return max_sum;
}
// Time complexity: O(n). Space complexity: O(1).



17.2 Compute the Levenshtein Distance

Write a program that takes two strings and computes
the minimum number of edits needed to transform the first
string into the second string.

int LevenshteinDistance( const string & A, const string & B ) {
	vector<vector<int>> distance_between_prefixes(A.size(), 
									vector<int>(B.size(), -1);
									
	return ComputeDistanceBetweenPrefixes(A, A.size() - 1, B, B.size() - 1,
											&distance_between_prefixes);	
}

int ComputeDistanceBetweenPrefixes(
	const string & A, int A_idx, const string & B, int B_idx,
	vector<vector<int>> * distance_between_prefixes) {
	
	vector<vector<int> & distance_between_prefixes = 
						* distance_between_prefixes_ptr;
						
	if (A_idx < 0) {   // if index of A is less than zero
		return B_idx + 1;
	}
	else if (B_idx < 0) { // if the index of B is less than zero
		return A_idx + 1;
	}
	
	if (distance_between_prefixes[A_idx][B_idx] == -1) {
		if (A[A_idx] == B[B_idx]) {
			distance_between_prefixes(A, A_idx - 1, B, B_idx - 1,
									distance_between_prefixes_ptr);
		}
		else {
			int substitute_last = ComputeDistanceBetweenPrefixes(
				A, A_idx - 1, B, B_idx - 1, distance_between_prefixes_ptr);
			int add_last = ComputeDistanceBetweenPrefixes(
				A, A_idx - 1, B, B_idx - 1, distance_between_prefixes_ptr);
			int delete_last = ComputeDistanceBetweenPrefixes(
				A, A_idx - 1, B, B_idx - 1, distance_between_prefixes_ptr);
			distance_between_prefixes[A_idx][B_idx] = 
				1 + min({substitute_last, add_last, delete_last});
		}
	}
	return distance_between_prefixes[A_idx][B_idx];	
}
// Time: O(ab)  Space: O(ab)



17.3 Count the number of ways to traverse a 2D array

Write a program that counts how many ways you can go
from the top-left to the bottom-right in a 2D array.

int NumberOfWays( int n, int m ) 
{	
	vector<vector<int>> number_of_ways(n, vector<in>(m, 0));
	return ComputeNumberOfWaysToXY( n - 1, m - 1, &number_of_ways);
}

int ComputeNumberOfWaysToXY( int x, int y, 
						vector<vector<int>> * number_of_ways_ptr) {
	if (x == 0 && y == 0) {
		return 1;
	}	
	vector<vector<int>> & number_of_ways = * number_of_ways_ptr;

	if (number_of_ways[x][y] == 0) {
	int ways_top = 
		x == 0 ? 0 : ComputeNumberOfWaysToXY(x - 1, y, number_of_ways_ptr);
	int ways_left = 
		y == 0 ? 0 : ComputeNumberOfWaysToXY(x, y - 1, number_of_ways_ptr);
	number_of_ways[x][y] = ways_top + ways_left;
	}
	return number_of_ways[x][y];
}

// Time: O(nm)  Space: O(nm)


17.12 Find the longest nondecreasing subsequence

Write a program that takes as input an array of numbers and
returns the length of a longest nondecreasing subsequence in
the array.

int LongestNondecreasingSubsequenceLength(const vector<int> & A) {
	
	vector<int> max_length(A.size(), 1);
	for (int i = 1; i < A.size(); ++i) {
		for (int j = 0; j < i; ++j) {
			if (A[i] >= A[j]) {
				max_length[i] = max(max_length[i], max_length[j] + 1);
			}
		}
	}
	return *max_element(max_length.begin(), max_length.end());
}
// Time: O(n^2)  Space: O(n)




///////////////////////////////////////////////////////////////////////
Greedy Algorithms
//////////////////////////////////////////////////////////////////////

A greedy algorithm is often the right choice for an optimization
problem where there's a natural set of choices to select from.

Write a program to make greedy choice for making change that results
in minimum number of coins.

int ChangeMaking( int cents ) {
	
	const array<int, 6> kCoins = {100, 50, 25, 10, 5, 1};
	int num_coins = 0;
	
	for (int i = 0; i < kCoins.size(); i++) {
		num_coins += cents / kCoins[i];
		cents %= kCoins[i];
	}
	return num_coins;
}
// Time: O(1)



18.1 Compute an optimum assignment of tasks

Design an algorithm that takes as input a set of tasks and 
returns an optimum assignment.

struct PairedTasks 
{
	int task_1, task_2;
}

vector<PairedTasks> OptimumTaskAssignment(vector<int> task_duration) 
{
	sort(task_durations.begin(), task_durations.end());
	vector<PairedTasks> optimum_assignments;
	
	for (int i = 0; j = task_durations.size() - 1; i < j; ++i, --j) 
	{
		optimum_assignments.emplace_back(
			PairedTasks{task_durations[i], task_durations[j]});
	}
	return optimum_assignments;
}
// Time: O( n log n)



18.2 Schedule to minimize waiting time

Given service times for a set of queries, compute a schedule for
processing the queries that minimizes the total waiting time.
Return the minimum waiting time.

int MinimumTotalWaitingTime( vector<int> service_times ) {
	// sort the service times in increasing order.
	sort(service_times.begin(), service_times.end());
	
	int total_waiting_time = 0;
	
	for (int i = 0; i < service_times.size(); ++i) {
		int num_remaining_queries = service_times.size() - (i + 1);
		total_waiting_time += service_times[i] * num_remaining_queries;
	}	
	return total_waiting_time;	
}
// Time: O( n log n)



/////////////////////////////////////////////////////////////////////
Invariant problems
/////////////////////////////////////////////////////////////////////

An invariant is a condition that is true during execution of a program


Write a program that takes as input a sorted and a given value and
determines if there are two entries in the array that add up to
that value. 
// method 1 if array is sorted

bool HasTwoSum( const vector<int> data, int target ) 
{
	int i = 0, j = data.size() - 1;
	
	while ( i <= j ) 
	{
		if ( data[i] + data[j] == target ) {
			return true;
		}
		else if ( data[i] + data[j] < target ) {
			++i;
		}
		else {
			--j;
		}
	}
	return false;
}

// Time: O(n), where n is the length of the array
// Space: O(1).

// Method 2 if the array is not sorted // two sum

bool TwoSum ( const vector<int> data, int sum ) {
	
	unordered_set<int> comp; // complements
	
	for (int value : data) {
		if (comp.find(value) != comp.end) {
			return true;
		}
		comp.add( sum - value );
	}
	return false;
}
// Time: O(n)



18.4 The 3-SUM problem

Design an algorithm that takes as input an array and a number,
and determines if there are three entries in teh array (not 
necessarily distinct) which add up to the specified number.

bool HasThreeSum( vector<int> data, int target ) {
	
	sort ( data.begin(), data.end() );
	
	for ( int value : data ) {
		
		if ( HasTwoSum( data, target - value ) {
			return true;
		}
	}
	return false;
}
// Time: O(n^2) 
// O( n log n ) time to sort and plus O(n) for the for loop!





18.7 Compute the maximum water trapped by a pair of vertical lines

Write a program which takes as input an integer array and returns
the pair of entries that trap the maximum amount of water.

int GetMaxTrappedWater ( const vector<int> heights ) 
{
	int i = 0; j = heights.size() - 1, max_water = 0;	
	while (i < j) {
		int width = j - i;
		max_water = max(max_water, width * min(heights[i], heights[j]));
		if (heights[i] > heights[j]) {
			--j;
		}
		else if (heights[i] < heights[j]) {
			++i;
		}
		else { // heights[i] == heights[j]
			++i, --j;
		}
	}
	return max_water;
}
// Time: O(n)



////////////////////////////////////////////////////////////////////
Graphs
////////////////////////////////////////////////////////////////////

Consider using a graph when you have to analyze any binary 
relationship, between objects, such as interlinked webpages, 
followers in a social graph, etc.

Some graph problems entail analyzing structure, looking for cycles
or connected components, DFS works particularly well for these
applications. 19.4

Some graph problems are related to optimization, find shortest
path from one vertex to another. BFS, Dijkstra's shortest path
algorithm, and minimum spanning tree are examples of graph algorithms
appropriate for optimization problems. 19.9

// Time complexity: 
// DFS: O( V + E )
// BFS: O( V + E )

// Space complexity:
// DFS: O( V )
// BFS: O( V )


Directed Acyclic Graph (DAG) is a directed graph in which there
are no cycles, i.e., paths which contain one or more edges and
which begin and endd at the same vertex.

Graphs are ideal for modeling and analyzing relationship between
pairs of objects. 

Graphs bootcamp problem:

Given a list of the outcomes of matches between pairs of teams, with
each outcome being a win or a loss. Given teams A and B, is there a
sequence of teams starting with A and ending with B such that each
team in the sequence has beaten the next team in the sequence?

// Depth first search DFS method!

struct MatchResult {
	string winning_team;
	string losing_team;
};

bool CanTeamABeatTeamB ( const vector<MatchResult> & matches,
						 const vector string & team_a, 
						 const vector string & team_b) {
	
	return IsReachableDFS(BuildGraph(matches), team_a, team_b,
						 make_unique<unordered_set<string>>().get());							 
}

unordered_map<string, unordered_set<string>> BuildGraph (
	const vector<MatchResult>& matches) {
	
	for (const MatchResult &match : matches) {
		graph[match.winning_team].emplace(match.losing_team);
	}
	return graph;	
}

bool IsReachableDFS( const unordered_map<string, unordered_set<string>> & graph,
					 const string & curr, const string & dest,
					 unordered_set<string> * visited_ptr) {
	unordered_set<string> & visited = * visited_ptr;
	if (curr == dest) {
		return true;
	}
	else if (visited.find(curr) != visited.end() ||
				graph.find(curr) == graph.end()) {
		return false;		
	}
	visited.emplace(curr);
	const auto& team_list = graph.at(curr);
	
	return any_of(begin(team_list), end(team_list), [&]
								(const string & team) {
		return IsReachableDFS(graph, team, dest, visited_ptr);		
	}
}
	// Time complexity: O(E)
	// Space: O(E), where E is the number of outcomes



19.1 Search maze
	
Given a 2D array of black and white entries representing a
maze with designated entrance and exit points, find a path
from the entrance to the exit, if one exists.

// DFS method!

typedef enum { WHITE, BLACK } Color;

struct Coordinate {
	int x, y;
	
	bool operator==(const Coordinate & that) const {
		return x == that.x && y == that.y;
	}	
};

vector<Coordinate> SearchMaze(vector<vector<Color>> maze, 
							  const Coordinates & start, 
							  const Coordinates & end) {
	// Create and initialize variables
	vector<Coordinate> path;
	maze[start.x][start.y] = BLACK;
	path.emplace_bac(start);
	if (!SearchMazeHelper(start, end, &maze, &path)) {
		path.pop_back();
	}
	return path;  // Empty path means no path between s and e		
}

// Perform DFS to find a feasible path

bool SearchMazeHelper(const Coordinate & cur, const Coordinate & end,
					  vector<vector<Color>> * maze, 
					  vector<Coordinate> * path) 
{
	if (cur == end) {
		return true; // if there is a path from start to end
	}
	
	const array<array<int, 2>, 4> kShift = {{{0, 1, {0, -1}, {1, 0, {-1, 0}}};
	for (const array<int, 2> & s : kShift) {
		Coordinate next {cur.x + s[0], cur.y + s[1]};
		if (IsFeasible(next, *maze)) {
			(*maze)[next.s][next.y] = BLACK;
			path->emplace_back(next);
			if (SearchMazeHelper(next, end, path)) {
				return true;
			}
			path->pop_back();
		}
	}
	return false;
}

// Checks cur is within maze and is a white pixel

bool IsFeasible(const Coordinate & cur, c
				const vector<vector<Color>> & maze) {
	return cur.x >= 0 && cur.x < max.size() && cur.y >= 0 &&
		   cur.y < maze[cur.x].size() && maze[cur.x][cur.y] == WHITE;					
}

// Time complexity for DFS search a maze is: O( |V| + |E| )




19.7 Transform one string to another

Given a dictionary D and two strings s and t, write a program to
determine if s produces t. Assume that all characters are lowercase
alphabets. If s does produce t, output the length of a shortest 
production sequence; otherwise, output - 1

Shortest paths in an undirected graph are naturally computed using
BFS

// Breadth First Search BFS method

int TransformString( unordered_set<string> D, const string & s,
					const string & t ) {
	
	struct StringWithDistance {
		string candidate_string;
		int distance;
	};
	// create a queue from above struct
	queue<stringWithDistance> q;		
	D.erase(s); // Mark s as visited by erasing it in D
	
	q.emplace(StringWithDistance{s, 0});
	
	while (!q.empty()) {
		StringWithDistance f(q.front());
		// Returns if we find a match
		if (f.candidate_string == t) {
			return f.distance; // Number of steps reaches t
		}
		
		// Tries all possible transformations of f.candidate_string
		string str = f.candidate_string;
		for (int i = 0; i < str.size(); ++i) {
			for (int j = 0; j < 26; ++j) { // iterates 'a' to 'z'
				str[i] = 'a' + j;
				auto it(D.find(str));				
				if ( it != D.end() ) {
					D.erase(it);
					q.emplace(StringWithDistance{str, f.distance + 1});
				}
			}
			str[i] = f.candidate_string[i]; // reverts the change of str
		}
		q.pop();
	}
	return -1;  // cannot find a possible transformations	
}
