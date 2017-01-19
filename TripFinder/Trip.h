// Trip.h

#include"Road.h"
#include<vector>
#include<cassert>
#include<iostream>

#ifndef PATH_LOCK
#define PATH_LOCK

class Trip
{
public:
	typedef Road::place_type	place_type;
	typedef std::vector<Road>	Roads;
	typedef Roads::size_type	size_type;

	Trip(void) : valid(false) {}
	Trip(place_type departFrom) : valid(true), departPlace(departFrom), xLength(0) {}

	bool isValid(void) const { return valid; }

	place_type arrivalPlace(void) const
	{
		assert(isValid());
		return (roads.size() == 0)
			? departPlace
			: roads[roads.size() - 1].arrivalPlace();
	}

	friend Trip operator+(const Trip &trip, const Road &road);

	void push_back(const Road &road);
	size_t length(void) { return xLength; }

	bool onTrip(place_type place) const;

	friend std::ostream &operator<<(std::ostream &stream, const Trip &trip);

private:
	bool		valid;
	place_type	departPlace;
	Roads		roads;
	size_t		xLength;
};

#endif
