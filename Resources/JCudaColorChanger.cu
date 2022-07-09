extern "C"
__global__ void computeColor(int width, int height, int paletteSize, int trailSize, int *palette,float *trailx, float *traily, int *rgb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = i % width;
    float y = i / height;
    float minRadius = (float) width;
    int point;
    for (point=0; point<trailSize; point++)
    {
        float dx = x - trailx[point];
        float dy = y - traily[point];
        minRadius = min(sqrt(dx * dx + dy * dy), minRadius);
    }
    int paletteIndex = min((int) minRadius, paletteSize);
    rgb[i] = palette[paletteIndex];
};