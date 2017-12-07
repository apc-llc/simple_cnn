#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer_t.h"

#pragma pack(push, 1)
struct relu_layer_t
{
	static void calc_grads( const tensor_t<float>& in, tensor_t<float>& grad_next_layer, void* layer )
	{
		((relu_layer_t*)layer)->calc_grads_( in, grad_next_layer );
	}

	static void fix_weights( const tensor_t<float>& in, void* layer )
	{

	}
	
	static void activate( const tensor_t<float>& in, void* layer )
	{
		((relu_layer_t*)layer)->activate_( in );
	}

	tensor_t<float> grads_in;
	tensor_t<float> out;

	relu_layer_t( tdsize in_size )
		:
		out( in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.x, in_size.y, in_size.z )
	{
	}

	void activate_( const tensor_t<float>& in )
	{
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					float v = in( i, j, z );
					if ( v < 0 )
						v = 0;
					out( i, j, z ) = v;
				}
	}

	void calc_grads_( const tensor_t<float>& in, tensor_t<float>& grad_next_layer )
	{
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					grads_in( i, j, z ) = (in( i, j, z ) < 0) ?
						(0) :
						(1 * grad_next_layer( i, j, z ));
				}
	}
};
#pragma pack(pop)

#endif // RELU_LAYER_H

