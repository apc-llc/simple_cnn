#include <iostream>
#include <fstream>
#include "cnn/cnn.h"

#define EPSILON 0.001

#define EXPECT_NEAR(val1, val2, abs_error) \
do \
{ \
  if (fabs(val1 - val2) > abs_error) \
  { \
    cout << "Incorrect control value: " << val1 << " != " << val2 << endl; \
    exit(1); \
  } \
} \
while (0)

using namespace std;

static uint32_t byteswap_uint32(uint32_t a)
{
	return ((((a >> 24) & 0xff) << 0) | (((a >> 16) & 0xff) << 8) |
		(((a >> 8) & 0xff) << 16) | (((a >> 0) & 0xff) << 24));
}

float train( layer_t* layers, int nlayers, tensor_t<float>& data, tensor_t<float>& expected )
{
	for ( int i = 0; i < nlayers; i++ )
	{
		if ( i == 0 )
			layers[i].activate( data );
		else
			layers[i].activate( layers[i - 1].out );
	}

	tensor_t<float> grads = layers[nlayers - 1].out - expected;

	for ( int i = nlayers - 1; i >= 0; i-- )
	{
		if ( i == nlayers - 1 )
			layers[i].calc_grads( grads );
		else
			layers[i].calc_grads( layers[i + 1].grads_in );
	}

	for ( int i = 0; i < nlayers; i++ )
	{
		layers[i].fix_weights();
	}

	float err = 0;
	for ( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ )
	{
		float f = expected.data[i];
		if ( f > 0.5 )
			err += fabs(grads.data[i]);
	}

	return err * 100;
}


void forward(layer_t* layers, int nlayers, tensor_t<float>& data )
{
	for ( int i = 0; i < nlayers; i++ )
	{
		if ( i == 0 )
			layers[i].activate( data );
		else
			layers[i].activate( layers[i - 1].out );
	}
}

struct case_t
{
	tensor_t<float> data;
	tensor_t<float> out;
	
	case_t() : data( 28, 28, 1 ), out( 10, 1, 1 ) { }
};

uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
	streamsize size = file.tellg();
	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	return buffer;
}

void* read_test_cases( case_t** cases_, int* ncases_ )
{
	uint8_t* train_image = read_file( "train-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "train-labels.idx1-ubyte" );

	uint32_t ncases = byteswap_uint32( *(uint32_t*)(train_image + 4) );
	case_t* cases = new case_t[ncases];

	for ( int i = 0; i < ncases; i++ )
	{
		case_t& c = cases[i];

		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;

		for ( int x = 0; x < 28; x++ )
			for ( int y = 0; y < 28; y++ )
				c.data( x, y, 0 ) = img[x + y * 28] / 255.f;

		for ( int b = 0; b < 10; b++ )
			c.out( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
	}

	delete[] train_image;
	delete[] train_labels;

	*cases_ = cases;
	*ncases_ = ncases;

	return cases;
}

int main()
{
	case_t* cases = NULL;
	int ncases = 0;
	read_test_cases( &cases, &ncases );

	conv_layer_t layer1( 1, 5, 8, cases[0].data.size ); // 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t layer2( layer1.out.size );
	pool_layer_t layer3( 2, 2, layer2.out.size );       // 24 * 24 * 8 -> 12 * 12 * 8
	fc_layer_t layer4(layer3.out.size, 10);             // 4 * 4 * 16 -> 10

	layer_t layers[] = { layer_t(&layer1), layer_t(&layer2), layer_t(&layer3), layer_t(&layer4) };
	int nlayers = sizeof(layers) / sizeof(layers[0]);

	float amse = 0;
	int ic = 0;

	for ( long ep = 0; ep < 100000; )
	{

		for ( int i = 0; i < ncases; i++ )
		{
			case_t& t = cases[i];
		
			float xerr = train( layers, nlayers, t.data, t.out );
			amse += xerr;

			ep++;
			ic++;

			if ( ep % 1000 == 0 )
			{
				const double err = amse / ic;
#if 1
				if (ep == 1000)
					EXPECT_NEAR(err, 76.7431, EPSILON);
				else if (ep == 2000)
					EXPECT_NEAR(err, 64.3735, EPSILON);
				else if (ep == 3000)
					EXPECT_NEAR(err, 55.8054, EPSILON);
				else if (ep == 4000)
					EXPECT_NEAR(err, 50.3824, EPSILON);
#endif
				cout << "case " << ep << " err = " << err << endl;
			}
		}
	}

	delete[] cases;

	while ( true )
	{
		uint8_t * data = read_file( "test.ppm" );

		if ( data )
		{
			uint8_t * usable = data;

			while ( *(uint32_t*)usable != 0x0A353532 )
				usable++;

#pragma pack(push, 1)
			struct RGB
			{
				uint8_t r, g, b;
			};
#pragma pack(pop)

			RGB * rgb = (RGB*)usable;

			tensor_t<float> image(28, 28, 1);
			for ( int i = 0; i < 28; i++ )
			{
				for ( int j = 0; j < 28; j++ )
				{
					RGB rgb_ij = rgb[i * 28 + j];
					image( j, i, 0 ) = (((float)rgb_ij.r
							     + rgb_ij.g
							     + rgb_ij.b)
							    / (3.0f*255.f));
				}
			}

			forward( layers, nlayers, image );
			tensor_t<float>& out = layers[nlayers - 1].out;
			for ( int i = 0; i < 10; i++ )
			{
				printf( "[%i] %f\n", i, out( i, 0, 0 )*100.0f );
			}

			delete[] data;
		}

		struct timespec wait;
		wait.tv_sec = 1;
		wait.tv_nsec = 0;
		nanosleep(&wait, nullptr);
	}

	return 0;
}

