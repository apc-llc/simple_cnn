#ifndef TENSOR_T_H
#define TENSOR_T_H

#include "point_t.h"
#include <cassert>
#include <vector>
#include <string.h>

template<typename T>
struct tensor_t
{
	T * data;

	tdsize size;

	tensor_t( int _x, int _y, int _z )
	{
		data = new T[_x * _y * _z];
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}

	tensor_t( const tensor_t& other )
	{
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * sizeof( T )
		);
		this->size = other.size;
	}

	T& operator()( int _x, int _y, int _z )
	{
		return this->get( _x, _y, _z );
	}

	const T& operator()( int _x, int _y, int _z ) const
	{
		return this->get( _x, _y, _z );
	}

	T& get( int _x, int _y, int _z )
	{
		assert( _x >= 0 && _y >= 0 && _z >= 0 );
		assert( _x < size.x && _y < size.y && _z < size.z );

		return data[
			_z * (size.x * size.y) +
				_y * (size.x) +
				_x
		];
	}

	const T& get( int _x, int _y, int _z ) const
	{
		assert( _x >= 0 && _y >= 0 && _z >= 0 );
		assert( _x < size.x && _y < size.y && _z < size.z );

		return data[
			_z * (size.x * size.y) +
				_y * (size.x) +
				_x
		];
	}

	~tensor_t()
	{
		delete[] data;
	}
};

#endif // TENSOR_T_H

