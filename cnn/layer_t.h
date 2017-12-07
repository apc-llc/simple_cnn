#ifndef LAYER_H
#define LAYER_H

#include "tensor_t.h"

typedef void (*calc_grads_t)( const tensor_t<float>& in, tensor_t<float>& grad_next_layer, void* layer );
typedef void (*fix_weights_t)( const tensor_t<float>& in, void* layer);
typedef void (*activate_t)( const tensor_t<float>& in, void* layer );

#pragma pack(push, 1)
struct layer_t
{
	tensor_t<float>& grads_in;
	tensor_t<float>& out;

	calc_grads_t calc_grads_underlying;
	fix_weights_t fix_weights_underlying;
	activate_t activate_underlying;
	
	void* underlying_layer;

	template<typename T>
	layer_t(T* layer) :
		underlying_layer((void*)layer),
		grads_in(layer->grads_in), out(layer->out),
		calc_grads_underlying(&T::calc_grads), fix_weights_underlying(&T::fix_weights), activate_underlying(&T::activate) { }

	void calc_grads( const tensor_t<float>& in, tensor_t<float>& grad_next_layer )
	{
		calc_grads_underlying( in, grad_next_layer, underlying_layer );
	}
	
	void fix_weights( const tensor_t<float>& in )
	{
		fix_weights_underlying( in, underlying_layer );
	}
	
	void activate( const tensor_t<float>& in )
	{
		activate_underlying( in, underlying_layer );
	}
};
#pragma pack(pop)

#endif // LAYER_H

