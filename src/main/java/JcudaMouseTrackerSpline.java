import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

import static java.lang.Math.min;
import static jcuda.driver.JCudaDriver.*;

/**
 * An example showing how to use the NVRTC (NVIDIA Runtime Compiler) API
 * to compile CUDA kernel code at runtime. We also add interpolation between mouseMovement events to create the
 * appearance of a worm.
 */
public class JcudaMouseTrackerSpline
{
    public static volatile Point mousePosition ; // to update the new position and paint accordingly
    public static volatile Point mousePosition2; // to send the Swing thread to sleep if the mouse is not moving

    public static void main(String[] args) throws IOException {
        ////////////// Set the problem constants and frames: /////////////

        // Set the frame
        JFrame frame = new JFrame("MouseTracker");

        // constants
        final int width = 1024;                                             // width of the problem
        final int height = width;
        final int paletteSize = 50;                                         // number of colors
        final int palette[] = new int[paletteSize + 1];                     // colors
        final int splineLength = 4;                                         // number of points used to create the Spline
        double splineX[] = new double[splineLength];                        // X positions for the spline
        double splineY[] = new double[splineLength];                        // Y positions for the spline
        final int trainSizeInterpolated = 1000;                             // number of mouse positions to store
        double trailxInterpolated[] = new double[trainSizeInterpolated];    // X positions
        double trailyInterpolated[] = new double[trainSizeInterpolated];    // Y positions
        int lastUpdatedInterpolatedPosition = 0;                            // Used to know where to update the trail.

        double [] timeVector = IntStream.range(0, splineLength).asDoubleStream().toArray();  // vector of times, used in the spline

        // Initialize palette values
        for (int i = 0; i < paletteSize; i++) {
            float h = i / (float) paletteSize;
            float b = 1.0f - h * h;
            palette[i] = Color.HSBtoRGB(h, 1.0f, b);
        }

        // Initialize trails outside the frame.
        Arrays.fill(trailxInterpolated,-width);
        Arrays.fill(trailyInterpolated,-width);

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

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputTrailX = new CUdeviceptr();
        cuMemAlloc(deviceInputTrailX, width*height * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputTrailX, Pointer.to(trailxInterpolated),
                trainSizeInterpolated * Sizeof.DOUBLE);

        CUdeviceptr deviceInputTrailY = new CUdeviceptr();
        cuMemAlloc(deviceInputTrailY, width*height * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputTrailY, Pointer.to(trailyInterpolated),
                trainSizeInterpolated * Sizeof.DOUBLE);

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
                /* //Uncomment to store an image in the closing
                try {
                    saveToFile(image);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                */
                // Clean up.
                cuMemFree(deviceInputPalette);
                cuMemFree(deviceInputTrailX);
                cuMemFree(deviceInputTrailY);
                System.exit(0);
            }
        });


        // Infinite loop that keeps the viewer alert to any mouse movement. Here we also compute the cubic spline curves
        // between the last detected mouse position
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

            //add to the fifo trails the mouse position the last position is in the las vector position spline needs this to be ordered
            JcudaMouseTrackerSpline.leftRotate(splineX, mousePosition.x);
            JcudaMouseTrackerSpline.leftRotate(splineY, mousePosition.y);

            //get the number of points to interpolate
            double[] timesToInterpolate = JcudaMouseTrackerSpline.computeTimesToInterpolate(splineX, splineY, splineLength);

            // Compute the splines for x and y in the given  "time" values
            double[] xInterpolated = Spline.cubicSplineSolver(timeVector, splineX, timesToInterpolate);
            double[] yInterpolated = Spline.cubicSplineSolver(timeVector, splineY, timesToInterpolate);

            // Update trails with the interpolated values
            updateTrailInterpolated(trailxInterpolated,xInterpolated ,trailyInterpolated, yInterpolated,
                    lastUpdatedInterpolatedPosition%trainSizeInterpolated);

            // Allocate the new trails to the GPU memory
            cuMemcpyHtoD(deviceInputTrailX, Pointer.to(trailxInterpolated),
                    trainSizeInterpolated * Sizeof.DOUBLE);
            cuMemcpyHtoD(deviceInputTrailY, Pointer.to(trailyInterpolated),
                    trainSizeInterpolated * Sizeof.DOUBLE);

            // Compute the new image colors
            colorKernel.runKernel(width, height, paletteSize, trainSizeInterpolated, deviceInputPalette, deviceInputTrailX,
                    deviceInputTrailY, deviceOutputRgb, imageRgb, viewer);

            // Increase counter
            lastUpdatedInterpolatedPosition+=xInterpolated.length;

            // Block the thread until new mouse event is triggered
            mousePosition2=mousePosition;
        }
    }

    /**
     * This function computes the points to resample between the last 2 sampled points. We don't want to change the
     * previous points, although the cubic spline is calculated using the last 4 points. We can assume this thanks to
     * the high frequency of the "mouse" triggering the events.
     * @param splineX Trail of X positions.
     * @param splineY Trail of Y positions
     * @param splineLength Length of the trail
     * @return double [] :the "times" where to interpolate. This is the number of intermediate points to interpolate between the
     * two last measured mouse positions.
     */
    private static double [] computeTimesToInterpolate(double[] splineX, double[] splineY, int splineLength){
        double lastDistanceXAbs = Math.abs(splineX[splineX.length-2] - splineX[splineX.length-1]);      //pixels x coord
        double lastDistanceYAbs = Math.abs(splineY[splineY.length-2] - splineY[splineY.length-1]);      //pixels y coord
        int maxDistance = Math.max((int) lastDistanceXAbs,(int)lastDistanceYAbs);                       //maximum pixels
        double[] timesToInterpolate = new double[maxDistance];                      //interpolation times in each pixel
        double space = 1. / maxDistance;                                            // time space between pixels
        int i = 0;
        while(i< maxDistance) {                                                     // get the actual time values
            timesToInterpolate[i] = splineLength - 2. + space*(i+1);
            i++;
        }
        return timesToInterpolate;
        // By changing the ratio of maxDistance we can reduce the number of points resampled i.e.: maxDistance/2
        // use the half points to interpolate 1 pixel yes, one pixel no... could be enough resolution.
    }

    /**
     * This function rotates to the left an array "appending" a value. This function is used to append a value to the
     * spline X/Y positions, because they must be ordered.
     * @param arr Array to rotate one position to the left
     * @param remplace Double with the value to append,
     */
    private static void leftRotate(double arr[], double remplace) {
        for (int i = 1; i < arr.length; i++) {
            arr[i - 1] = arr[i];
        }
        arr[arr.length-1] = remplace;
    }

    /**
     * This function updates simultaneously the values of the trails of the interpolated arrays.
     * @param trailInterpolatedX Array of the old interpolated X values.
     * @param interpolatedXValues Array with the new interpolated X values to replace the oldest ones in the old array.
     * @param trailInterpolatedY Array of the old interpolated Y values.
     * @param interpolatedYValues Array with the new interpolated Y values to replace the oldest ones in the old array.
     * @param firstIdx Index where to start adding the new values.
     */
    private static void updateTrailInterpolated(double[] trailInterpolatedX, double [] interpolatedXValues,
                                               double[] trailInterpolatedY, double [] interpolatedYValues,int firstIdx){
        int shift= Math.max(0,interpolatedXValues.length-trailInterpolatedX.length);    // Shift to the right if the new
                                                                    //interpolated values are more that the trail length
        int realFirstIdx =firstIdx+shift;   // Real first index to use from the new interpolated values
        IntStream.range(shift, interpolatedXValues.length).parallel().forEach(index -> {    // replace
            int interpolatedIdx=(realFirstIdx + index )% trailInterpolatedX.length;     //index of the parallel update
            trailInterpolatedX[interpolatedIdx] = interpolatedXValues[index];
            trailInterpolatedY[interpolatedIdx] = interpolatedYValues[index];
        });
    }

    private static void saveToFile (BufferedImage img) throws IOException {
        ImageIO.write(img, "png", new File("out/image/Frame_Image_Spline.png"));
    }
}

