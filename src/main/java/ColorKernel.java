import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

import javax.swing.*;

import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.nvrtc.JNvrtc.*;

public class ColorKernel {

    int gridSizeX=1;
    int gridSizeY=1;
    int gridSizeZ=1;
    int blockSizeX=1;
    int blockSizeY=1;
    int blockSizeZ=1;
    int sharedMemBytes=0;
    CUstream hStream=null;
    Pointer extra=null;
    CUfunction function;

    /**
     * This function sets the kernel and creates the function that is going to be executed in the GPU.
     * It reads a Cuda file "example.cu".
     * IMPORTANT!!!!! We Be careful with the path. By default, is referred to resources.
     * @param cudaFileName : cuda filename string "example.cu".
     * @param functionName : Name of the function, tipically the same as in the .cu file function.
     * @throws IOException
     */
    public ColorKernel(String cudaFileName, String functionName) throws IOException {
        this.function = computeKernelFunction(cudaFileName,functionName);
    }

    /**
     * This function fills the Kernel with grid/block parameters to know threads and block executing those threads
     * @param gridSizeX
     * @param gridSizeY
     * @param gridSizeZ
     * @param blockSizeX
     * @param blockSizeY
     * @param blockSizeZ
     * @param sharedMemBytes
     * @param hStream
     * @param extra
     */
    public void fillKernel(int gridSizeX, int gridSizeY, int gridSizeZ, int blockSizeX, int blockSizeY, int blockSizeZ,
                  int sharedMemBytes, CUstream hStream, Pointer extra) {
        this.gridSizeX = gridSizeX;
        this.gridSizeY = gridSizeY;
        this.gridSizeZ = gridSizeZ;
        this.blockSizeX = blockSizeX;
        this.blockSizeY = blockSizeY;
        this.blockSizeZ = blockSizeZ;
        this.sharedMemBytes = sharedMemBytes;
        this.hStream = hStream;
        this.extra = extra;
    }

    public static CUfunction computeKernelFunction(String cudaFileName, String functionName) throws IOException {
        String programSourceCode = new String(JcudaMouseTracker.class.getClassLoader().getResourceAsStream(cudaFileName).readAllBytes());
        // Retrieve program source code as string
        //new String(Files.readAllBytes(Paths.get(new File("").getAbsolutePath()+"/resources/"+cudaFileName)));
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);


        // Use the NVRTC to create a program by compiling the source code
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(
                program, programSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);

        // Print the compilation log (for the case there are any warnings)
        String programLog[] = new String[1];
        nvrtcGetProgramLog(program, programLog);

        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, functionName);
        return function;
    }

    public  void runKernel(int width, int height, int paletteSize, int trailSize, CUdeviceptr deviceInputPalette,
                           CUdeviceptr deviceInputTrailX, CUdeviceptr deviceInputTrailY, CUdeviceptr deviceOutputRgb,
                           int[] imageRgb, JComponent viewer){

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{width}),
                Pointer.to(new int[]{height}),
                Pointer.to(new int[]{paletteSize}),
                Pointer.to(new int[]{trailSize}),
                Pointer.to(deviceInputPalette),
                Pointer.to(deviceInputTrailX),
                Pointer.to(deviceInputTrailY),
                Pointer.to(deviceOutputRgb)
        );

        // Call the kernel function, which was obtained from the
        // module that was compiled at runtime
        cuLaunchKernel(function,
                gridSizeX,  gridSizeY, gridSizeZ,      // Grid dimension
                blockSizeX, blockSizeY, blockSizeZ,      // Block dimension
                sharedMemBytes, hStream,               // Shared memory size and stream
                kernelParams, extra // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        int hostOutput[] = new int[width*height];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutputRgb,
                hostOutput.length * Sizeof.INT);
        System.arraycopy(hostOutput, 0, imageRgb, 0, imageRgb.length);
        viewer.repaint();
    }

}
