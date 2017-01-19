// Road.h

#include"Tools.h"
#include<cassert>
#include<string>
using namespace tools_namespace;

#ifndef PATHSEGMENT_LOCK
#define PATHSEGMENT_LOCK

class Road
{
public:
	typedef std::string name_type;
	typedef std::string place_type;

	Road(name_type name, place_type departurePlace, place_type arrivalPlace)
		: roadName(name), departPlace(departurePlace), arrivePlace(arrivalPlace)
	{}

	name_type	name(void)				const { return roadName;    }
	place_type	departurePlace(void)	const { return departPlace; }
	place_type  arrivalPlace(void)		const { return arrivePlace; }

	bool hasEndPoint(place_type endPoint) const
	{ return noCaseEQ(endPoint, departPlace) || noCaseEQ(endPoint, arrivePlace); }

	Road departingFrom(place_type desiredDeparturePlace) const
	{
		assert(hasEndPoint(desiredDeparturePlace));
		return noCaseEQ(desiredDeparturePlace, departPlace)
			? *this
			: Road(roadName, arrivePlace, departPlace);
	}

private:
	name_type roadName;
	place_type departPlace;
	place_type arrivePlace;
};

#endif