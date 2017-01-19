// TripFinder.h

#include"Road.h"
#include"Trip.h"
#include<vector>

#ifndef TRIPFINDER_LOCK
#define TRIPFINDER_LOCK

// For keeping track of all places, and shortest path to each
class PlaceInfo
{
public:
	PlaceInfo(Road::place_type i_place) : place(i_place), found(false) {}

	Road::place_type place;
	bool found;
	size_t shortestPathLength;
};

///////////////////////////////////////////////////////////////////////////////////////

class TripFinder
{
public:
	static const char ROAD_INPUT_DELIMITER = '\t';

	typedef Road::place_type		place_type;
	typedef std::vector<Road>		Roads;

	typedef std::vector<PlaceInfo>	Places;

	TripFinder(void) { getRoads(); }

	Trip findTrip(place_type departurePlace, place_type arrivalPlace, bool pruneSearch);

	place_type getPlace(std::string prompt);

private:
	Places	places;
	Roads	roads;

	void getRoads(void);

	void search
		(
		Trip				currentTrip,
		Trip				&bestTripSoFar,
		const place_type	&tripArrivalPlace,
		bool				pruneSearch
		);

	size_t find(place_type place);
	bool isKnown(place_type place);
	void insertUnique(place_type place);
};


#endif