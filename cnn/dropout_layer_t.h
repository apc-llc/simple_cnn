#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer_t.h"

#pragma pack(push, 1)
struct dropout_layer_t
{
	static void calc_grads( const tensor_t<float>& in, tensor_t<float>& grad_next_layer, void* layer )
	{
		((dropout_layer_t*)layer)->calc_grads_( in, grad_next_layer );
	}

	static void fix_weights( const tensor_t<float>& in, void* layer )
	{

	}
	
	static void activate( const tensor_t<float>& in, void* layer )
	{
		((dropout_layer_t*)layer)->activate_( in );
	}

	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<bool> hitmap;
	float p_activation;

	dropout_layer_t( tdsize in_size, float p_activation )
		:
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x, in_size.y, in_size.z ),
		hitmap( in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.x, in_size.y, in_size.z ),
		p_activation( p_activation )
	{
		
	}

	void activate_( const tensor_t<float>& in )
	{
		this->in = in;

		for ( int i = 0; i < in.size.x*in.size.y*in.size.z; i++ )
		{
			bool active = (rand() % RAND_MAX) / float( RAND_MAX ) <= p_activation;
			hitmap.data[i] = active;
			out.data[i] = active ? in.data[i] : 0.0f;
		}
	}

	void calc_grads_( const tensor_t<float>& in, tensor_t<float>& grad_next_layer )
	{
		for ( int i = 0; i < in.size.x*in.size.y*in.size.z; i++ )
			grads_in.data[i] = hitmap.data[i] ? grad_next_layer.data[i] : 0.0f;
	}
};
#pragma pack(pop)

#endif // DROPOUT_LAYER_H

