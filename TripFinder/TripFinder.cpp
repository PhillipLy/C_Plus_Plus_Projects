// TripFinder.cpp

#include"TripFinder.h"
#include"Road.h"
#include"Trip.h"
#include"Tools.h"
#include<fstream>
#include<iostream>
#include<cassert>
using namespace tools_namespace;
using namespace std;


void TripFinder::getRoads(void)
{
	ifstream roadFile;
	openIfStream(roadFile, "Enter name/trip of file containing road information: ");
	while (true)
	{
		Road::name_type	roadName;
		place_type		departurePlace, arrivalPlace;
		if (getline(roadFile, roadName, ROAD_INPUT_DELIMITER).fail()
			|| getline(roadFile, departurePlace, ROAD_INPUT_DELIMITER).fail()
			|| getline(roadFile, arrivalPlace).fail())
			break;
		roads.push_back(Road(roadName, departurePlace, arrivalPlace));
		insertUnique(departurePlace);
		insertUnique(arrivalPlace);
	}
	cout << roads.size() << " roads loaded." << endl;
	cout << places.size() << " cities registered." << endl;
}

TripFinder::place_type TripFinder::getPlace(string prompt)
{
	while (true)
	{
		place_type place = getLine(prompt);
		if (isKnown(place)) return place;
		cout << "Unknown place: Try again." << endl;
	}
}

Trip TripFinder::findTrip
(place_type tripDeparturePlace, place_type tripArrivalPlace, bool pruneSearch)
{
	// trivial trip
	if (noCaseEQ(tripDeparturePlace, tripArrivalPlace))
		return Trip(tripArrivalPlace);

	Trip bestSoFar; // initially not valid

	// begin search from an empty trip
	// successively save better trips as found 
	search(Trip(tripDeparturePlace), bestSoFar, tripArrivalPlace, pruneSearch);

	return bestSoFar;
}

void TripFinder::search
(
Trip				currentTrip,
Trip				&bestTripSoFar,
const place_type	&tripArrivalPlace,
bool				pruneSearch
)
{
	// extend trip with all trip roads that fit
	for (vector<Road>::size_type roadIndex = 0; roadIndex < roads.size(); ++roadIndex)
	{
		// try extending trip with this road
		Trip newTrip = currentTrip + roads[roadIndex];
		if (!newTrip.isValid()) continue;

		// if solution found,
		if (noCaseEQ(newTrip.arrivalPlace(), tripArrivalPlace))
		{
			// see if it's better than previously found solutions
			if (!bestTripSoFar.isValid() || newTrip.length() < bestTripSoFar.length())
				bestTripSoFar = newTrip;
			continue;
		}

		if (pruneSearch)
		{
			PlaceInfo &data = places[find(newTrip.arrivalPlace())]; // shouldnt fail
			if (data.found && newTrip.length() >= data.shortestPathLength)
				continue; // prune - no point in searching
			else
			{
				data.found = true;
				data.shortestPathLength = newTrip.length();
			}
		}

		search(newTrip, bestTripSoFar, tripArrivalPlace, pruneSearch);
	}
}

// use an ordered list to optimize search (binary)

// returns size if not found
size_t TripFinder::find(place_type place)
{
	if (places.size() == 0) return places.size();

	place = toLower(place);

	Places::size_type lowBound = 0;
	Places::size_type highBound = places.size() - 1;
	while (lowBound < highBound)
	{
		Places::size_type middle = (lowBound + highBound) / 2;
		if (place <= places[middle].place)	highBound = middle;
		else									lowBound = middle + 1;
	}
	return (place == places[lowBound].place) ? lowBound : places.size();
}

bool TripFinder::isKnown(place_type place)
{
	return places.size() > find(place);
}

void TripFinder::insertUnique(place_type place)
{
	if (isKnown(place)) return;

	place = toLower(place);

	// make a hole
	Places::size_type holeLocation = places.size();
	places.push_back(PlaceInfo("")); // goes into location above

	// move hole to ordered insertion point
	for (; holeLocation > 0 && places[holeLocation - 1].place > place; --holeLocation)
		places[holeLocation] = places[holeLocation - 1];

	// insert
	places[holeLocation] = PlaceInfo(place);
}
