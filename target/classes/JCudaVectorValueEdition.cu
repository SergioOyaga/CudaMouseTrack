extern "C"
__global__ void changeVectorValue(int position, double value,double *vector)
{
    vector[position]=value;
};