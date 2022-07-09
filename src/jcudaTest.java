import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;

import static java.lang.Math.min;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;

/**
 * An example showing how to use the NVRTC (NVIDIA Runtime Compiler) API
 * to compile CUDA kernel code at runtime.
 */
public class jcudaTest
{
    public static volatile Point mousePosition = new Point(0,0); // to update the new position and paint accordingly
    public static volatile Point mousePosition2 =  new Point(0,0); // to send the Swing thread to sleep if the mouse is not moving

    public static void main(String[] args) throws IOException {
        ////////////// Set the problem constants and frames: /////////////

        // Set the frame
        JFrame frame = new JFrame("MouseTracker");

        // constants
        final int width = 1024;                             // width of the problem
        final int height = width;
        final int paletteSize = 50;                        // number of colors
        final int palette[] = new int[paletteSize + 1];     // colors
        final int trailSize = 100;                          // number of mouse positions to store
        float trailx[] = new float[trailSize];              // X positions
        float traily[] = new float[trailSize];              // Y positions
        int trailLastUpdatedPosition = 0;

        // Initialize palette values
        for (int i = 0; i < paletteSize; i++) {
            float h = i / (float) paletteSize;
            float b = 1.0f - h * h;
            palette[i] = Color.HSBtoRGB(h, 1.0f, b);
        }

        // Create the image.
        final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        // Override the paint handler to just copy the image, the viewer is where the image is displayed
        JComponent viewer = new JComponent(){
            @Override public void paintComponent(Graphics g) {
                g.drawImage(image, 0, 0, width, height, this);
            }
        };

        // Set the size of JComponent which displays the image
        viewer.setPreferredSize(new Dimension(width, height));

        // Swing housework to create the frame
        frame.getContentPane().add(viewer);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        //The image as array
        final int[] imageRgb = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();

        // We use this to synchronize access from Swing display thread to the GPU have a valid context
        final Object doorBell = new Object();

        //Instantiate the kernel creating the function out of the .cu file with the name of the C function.
        ColorKernel colorKernel = new ColorKernel("JCudaColorChanger.cu","computeColor");
        VectorKernel vectorXKernel = new VectorKernel("JCudaVectorValueEdition.cu", "changeVectorValue");
        VectorKernel vectorYKernel = new VectorKernel("JCudaVectorValueEdition.cu", "changeVectorValue");

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputTrailX = new CUdeviceptr();
        cuMemAlloc(deviceInputTrailX, trailSize * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputTrailX, Pointer.to(trailx),
                trailSize * Sizeof.FLOAT);

        CUdeviceptr deviceInputTrailY = new CUdeviceptr();
        cuMemAlloc(deviceInputTrailY, trailSize * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputTrailY, Pointer.to(traily),
                trailSize * Sizeof.FLOAT);

        CUdeviceptr deviceInputPalette = new CUdeviceptr();
        cuMemAlloc(deviceInputPalette, palette.length * Sizeof.INT);
        cuMemcpyHtoD(deviceInputPalette, Pointer.to(palette),
                palette.length * Sizeof.INT);

        // Allocate device output memory
        CUdeviceptr deviceOutputRgb = new CUdeviceptr();
        cuMemAlloc(deviceOutputRgb, imageRgb.length * Sizeof.INT);

        // Compute the grids/blocks sizes for the problem.
        int blockSizeX = min(width,1024);
        int gridSizeX = min((width*height+blockSizeX-1)/blockSizeX,1024);

        // Fill the kernel class with all the information
        colorKernel.fillKernel(gridSizeX,1,1,blockSizeX,1,1,0,
                null,null);

        vectorXKernel.fillKernel(1,1,1,1,1,1,0,
                null,null);
        vectorYKernel.fillKernel(1,1,1,1,1,1,0,
                null,null);

        // Mouse listener which collects the latest the mouse position from Mandelbrot view whenever the mouse is moved
        viewer.addMouseMotionListener(new MouseMotionAdapter(){
            @Override public void mouseMoved(MouseEvent e) {
                mousePosition = e.getPoint();           // mouse position

                // tell the waiting Swing thread that we have a new position
                synchronized (doorBell) {
                    doorBell.notify();
                }
            }
        });

        // Window listener to dispose Kernel resources on user exit
        frame.addWindowListener(new WindowAdapter(){
            public void windowClosing(WindowEvent _windowEvent) {
                // Clean up.
                cuMemFree(deviceInputPalette);
                cuMemFree(deviceInputTrailX);
                cuMemFree(deviceInputTrailY);
                System.exit(0);
            }
        });


        // Infinite loop that keeps the viewer alert to any mouse movement.
        while (true) {
            /** Wait for the user to move mouse **/
            while (mousePosition == mousePosition2) {
                synchronized (doorBell) {
                    try {
                        doorBell.wait();
                    } catch (InterruptedException ie) {
                        ie.getStackTrace();
                    }
                }
            }
            vectorXKernel.runKernel(trailLastUpdatedPosition % trailx.length, mousePosition.x, deviceInputTrailX);
            vectorYKernel.runKernel(trailLastUpdatedPosition % traily.length, mousePosition.y, deviceInputTrailY);
            colorKernel.runKernel(width, height, paletteSize, trailSize, deviceInputPalette, deviceInputTrailX,
                    deviceInputTrailY, deviceOutputRgb, imageRgb, viewer);
            trailLastUpdatedPosition++;
            mousePosition2=mousePosition;
        }
    }
}
