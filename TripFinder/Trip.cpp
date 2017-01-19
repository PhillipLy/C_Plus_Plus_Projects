// Trip.cpp

#include"Trip.h"
#include"Road.h"
#include<iostream>
#include<cassert>
using namespace std;


bool Trip::onTrip(place_type place) const
{
	for (size_type search = 0; search < roads.size(); ++search)
	if (noCaseEQ(place, roads[search].arrivalPlace()))
		return true;
	return isValid() && noCaseEQ(place, departPlace);
}

void Trip::push_back(const Road &road)
{
	if (onTrip(road.arrivalPlace())) valid = false;
	else
	{
		roads.push_back(road);
		++xLength;
	}
}

std::ostream &operator<<(std::ostream &stream, const Trip &trip)
{
	if (!trip.isValid())
		stream << "There is no trip." << endl;

	else if (trip.roads.size() == 0)
		stream << "You're already there!" << endl;

	else for (Trip::size_type index = 0; index < trip.roads.size(); ++index)
		stream << "Take " << trip.roads[index].name()
		<< " from " << trip.roads[index].departurePlace()
		<< " to " << trip.roads[index].arrivalPlace()
		<< '.' << endl;
	return stream;
}

Trip operator+(const Trip &trip, const Road &road)
{
	if (!trip.isValid()
		|| !road.hasEndPoint(trip.arrivalPlace()))
		return Trip();

	Trip newTrip(trip);
	newTrip.push_back(road.departingFrom(trip.arrivalPlace()));
	return newTrip;
}