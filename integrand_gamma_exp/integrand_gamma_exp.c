#include <math.h>

const double sqrt2 = 1.4142135623730951;

double integrand_gamma_exp(int n, double *x, void *user_data)
{
        double freq_r = *(double *)user_data;
        double gamma = *((double *)user_data + 1);
        double a = *((double *)user_data + 2);
        double xx=x[0]*x[0], yy=x[1]*x[1];

        return exp(-pow(a*sqrt(xx+yy), gamma)) * cos(freq_r/sqrt2*(x[0]+x[1]));
}
