extern "C"
__global__ void computeColor(int width, int height, int paletteSize, int trailSize, int *palette,double *trailx, double *traily, int *rgb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double x = i % width;
    double y = i / height;
    double minRadius = (double) width;
    int point;
    for (point=0; point<trailSize; point++)
    {
        double dx = x - trailx[point];
        double dy = y - traily[point];
        minRadius = min(sqrt(dx * dx + dy * dy), minRadius);
    }
    int paletteIndex = min((int) minRadius, paletteSize);
    rgb[i] = palette[paletteIndex];
};