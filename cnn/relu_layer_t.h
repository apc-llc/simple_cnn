#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer_t.h"

#pragma pack(push, 1)
struct relu_layer_t
{
	static void calc_grads( tensor_t<float>& grad_next_layer, void* layer )
	{
		((relu_layer_t*)layer)->calc_grads_(grad_next_layer);
	}

	static void fix_weights( void* layer )
	{
		((relu_layer_t*)layer)->fix_weights_();
	}
	
	static void activate( tensor_t<float>& in, void* layer )
	{
		((relu_layer_t*)layer)->activate_( in );
	}

	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;

	relu_layer_t( tdsize in_size )
		:
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.x, in_size.y, in_size.z )
	{
	}

	void activate_( tensor_t<float>& in )
	{
		this->in = in;
		activate_();
	}

	void activate_()
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

	void fix_weights_()
	{

	}

	void calc_grads_( tensor_t<float>& grad_next_layer )
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

