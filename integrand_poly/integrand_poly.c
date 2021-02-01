#include <math.h>

const double sqrt2 = 1.4142135623730951;
#define MAX(a, b) ((a) > (b) ? (a) : (b))

double integrand_poly(int n, double *x, void *user_data)
{
        double freq_r = *(double *)user_data;
        double order = *((double *)user_data + 1);
        double xx=x[0]*x[0], yy=x[1]*x[1];

        return pow(MAX(0., 1-sqrt(xx+yy)), order) * cos(freq_r/sqrt2*(x[0]+x[1]));
}
