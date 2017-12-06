#ifndef GRADIENT_H
#define GRADIENT_H

struct gradient_t
{
	float grad;
	float oldgrad;
	gradient_t()
	{
		grad = 0;
		oldgrad = 0;
	}
};

#endif // GRADIENT_H

