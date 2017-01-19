// Main.cpp
// Trip Finder - Phillip Ly
// Use recursion to find the shortest trip between cities


#include"TripFinder.h"
#include"Tools.h"
#include<iostream>
#include<time.h>
using namespace tools_namespace;
using namespace std;

void test(TripFinder &tripFinder);

int main(void)
{
	TripFinder tripFinder;

	do
	{
		test(tripFinder);
		cout << endl;
	} while (getBool("Do you want to find another trip? "));

	return 0;
}

void test(TripFinder &tripFinder)
{
	typedef TripFinder::place_type place_type;
	place_type departurePlace = tripFinder.getPlace("Enter departure city in desired trip: ");
	place_type arrivalPlace = tripFinder.getPlace("Enter arrival city in desired trip:   ");

	bool pruneSearch = getBool("Do you want a faster/intelligent search? ");

	time_t startTime = time(0);

	cout << tripFinder.findTrip(departurePlace, arrivalPlace, pruneSearch);

	double elapsedMinutes = (time(0) - startTime) / 60.0;
	cout << "Elapsed time: " << elapsedMinutes << " minutes." << endl;
}
