import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;

public class VectorKernel {
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

    public VectorKernel(String cudaFileName, String functionName) throws IOException {
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

    /**
     * This function sets the kernel and creates the function that is going to be executed in the GPU.
     * It reads a Cuda file "example.cu".
     * IMPORTANT!!!!! We Be careful with the path. By default, is referred to resources.
     * @param cudaFileName : cuda filename string "example.cu".
     * @param functionName : Name of the function, tipically the same as in the .cu file function.
     * @return a CUfunctionthat refers to the function that has to be executed in GPU.
     * @throws IOException
     */
    public static CUfunction computeKernelFunction(String cudaFileName, String functionName) throws IOException {
        String programSourceCode = new String(jcudaTest.class.getClassLoader().getResourceAsStream(cudaFileName).readAllBytes());

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

    public  void runKernel(int position, float value, CUdeviceptr deviceInput ){

        Pointer vectorXKernelParameters = Pointer.to(
                Pointer.to(new int[]{position}),
                Pointer.to(new float []{value}),
                Pointer.to(deviceInput)
        );

        // Call the kernel function, which was obtained from the
        // module that was compiled at runtime
        cuLaunchKernel(function,
                gridSizeX,  gridSizeY, gridSizeZ,      // Grid dimension
                blockSizeX, blockSizeY, blockSizeZ,      // Block dimension
                sharedMemBytes, hStream,               // Shared memory size and stream
                vectorXKernelParameters, extra // Kernel- and extra parameters
        );
        cuCtxSynchronize();
    }

}
