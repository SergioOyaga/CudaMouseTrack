extern "C"
__global__ void changeVectorValue(int position, float value,float *vector)
{
    vector[position]=value;
};