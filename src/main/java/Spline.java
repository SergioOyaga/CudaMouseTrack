/**
 * This class contains the necessary information to interpolate the points od any set of points using cubic Splines.
 * Allows to directly interpolate using a static method tha computes all the spline parameters, of you can create a
 * class that computes all the parameters and after allow you to interpolate a new value or lis of values. This is
 * problem dependant.
 */
public class Spline{
    private final double[] xInputValues;
    private final double[] yInputValues;
    private double [] dVec;
    private double[] c2;
    private double[] c3;
    private double[] delta;
    private double[] del1;
    private double[] del2;
    private double[] h;
    private final int numberOfKnots;

    /**
     * Class constructor, initializes all values from two list f arrays. X must be ordered.
     * @param xInputValues X values as an array of doubles (double []). It must be ordered.
     * @param yInputValues X values as an array of doubles (double []). It must be ordered.
     */
    public Spline(double[] xInputValues, double[] yInputValues) {

        assert xInputValues.length==yInputValues.length : "Sizes of knots not matching: x.length="+ xInputValues.length +" , y.length="+yInputValues.length;
        assert xInputValues.length>1 : "Can not create an interpolation from one point. x.length="+ xInputValues.length;
        for (int i = 1; i < xInputValues.length; i++) {
            assert (xInputValues[i - 1] <= xInputValues[i]): "The xInputValues must be ordered from min to max. [..., "+ xInputValues[i - 1]+", "+ xInputValues[i]+", ...]";
        }
        this.xInputValues = xInputValues;
        this.yInputValues = yInputValues;
        this.numberOfKnots = xInputValues.length;
        this.computeSplineParams();
    }

    /**
     * Computes the Spline params.
     */
    private void computeSplineParams(){
        this.dVec = new double [this.numberOfKnots];                  // Array of derivatives at each of the xInputValues points
        //fixed params dependent on the problem
        this.c2 = new double [numberOfKnots];
        this.c3 = new double [numberOfKnots];
        this. delta = new double [numberOfKnots];
        this.del1 = new double [numberOfKnots];
        this. del2 = new double [numberOfKnots];
        this. h = new double [numberOfKnots];


        double [] wk = new double[this.numberOfKnots];          // wk is a 2*M work array, used in the calculation of the derivatives at each of the entered x-values
        int nj;                                    //Array indices

        double dummy, tempx;                       //Dummy variables

        // Differences between the x-values are stored in wk[0][. . .], starting with h[1]
        // Divided differences between y-values are stored in wk[. . .], starting with wk[1]
        // Note that h[0] and wk[0] are not filled in this loop; they are presently left unassigned
        nj = this.numberOfKnots - 1;
        for (int i = 0; i < nj; ++i){
            this.h [i+1] = this.xInputValues[i+1] - this.xInputValues[i];
            wk[i+1] = (this.yInputValues[i+1] - this.yInputValues[i])/this.h[i+1];
        }
        if (this.numberOfKnots == 2){
            wk[0] = this.h[0] = 1.0;
            this.dVec[0] = this.dVec[1] = wk[1];
            this.dVec[0] *= 2.0;
        }
        else {
            tempx = dummy = this.h[1];
            wk[0] = this.h[2];
            this.h[0] = tempx + this.h[2];
            dummy *= dummy*wk[2];
            this.dVec[0] = ((tempx + 2.0*this.h[0])*wk[1]*this.h[2] + dummy)/this.h[0];

            for (var i = 1; i < nj; ++i){
                tempx = -(this.h[i+1]/wk[i-1]);
                this.dVec[i] = tempx*this.dVec[i-1] + 3.0*(this.h[i]*wk[i+1] + this.h[i+1]*wk[i]);
                wk[i] = tempx*this.h[i-1] + 2.0*(this.h[i] + this.h[i+1]);
            }

            if (this.numberOfKnots == 3){
                this.dVec[2] = 2.0*wk[2];
                wk[2] = 1.0;
                tempx = -(1.0/wk[1]);
            }
            else {
                tempx = this.h[nj-1] + this.h[nj];
                dummy = this.h[nj]*this.h[nj]*(this.yInputValues[nj-1] - this.yInputValues[nj-2]);
                dummy /= this.h[nj-1];
                this.dVec[nj] = ((this.h[nj] + 2.0*tempx)*wk[nj]*this.h[nj-1] + dummy)/tempx;
                tempx = -(tempx/wk[nj-1]);
                wk[nj] = this.h[nj-1];
            }
            // Complete forward pass of Gauss Elimination
            wk[nj] = tempx*this.h[nj-1] + wk[nj];
            this.dVec[nj] = (tempx*this.dVec[nj-1] + this.dVec[nj])/wk[nj];

        }

        //Carry out back substitution
        for (var i= nj-1; i >= 0; i--){
            this.dVec[i] = (this.dVec[i] - this.h[i]*this.dVec[i+1])/wk[i];
        }

        for ( int knotVectorIdx=1; knotVectorIdx<this.numberOfKnots;knotVectorIdx++){
            this.delta[knotVectorIdx] = (this.yInputValues[knotVectorIdx] - this.yInputValues[knotVectorIdx - 1])/this.h[knotVectorIdx];
            this.del1[knotVectorIdx]  = (this.dVec[knotVectorIdx - 1] - this.delta[knotVectorIdx])/this.h[knotVectorIdx];
            this.del2[knotVectorIdx]  = (this.dVec[knotVectorIdx] - this.delta[knotVectorIdx])/this.h[knotVectorIdx];

            this.c2[knotVectorIdx]  = -(this.del1[knotVectorIdx] + this.del1[knotVectorIdx] + this.del2[knotVectorIdx]);
            this.c3[knotVectorIdx]  = (this.del1[knotVectorIdx] + this.del2[knotVectorIdx])/this.h[knotVectorIdx];
        }
    }

    /**
     * Computes the interpolation (y values) af an array of values (x values) for a given spline interpolation
     * (object of this class).
     * @param xInputToInterpolateValues Array if x values to interpolate (double []).
     * @return an array of y values (double []) corresponding to the interpolation of the given x values.
     */
    public double[] cubicSplineSolver( double []xInputToInterpolateValues){
        for (double xInter:xInputToInterpolateValues){
            assert xInputValues[0]<=xInter && xInter<= xInputValues[xInputValues.length-1]: " There are values to extrapolate. This function only interpolates values.";
        }

        int numberToInterpolate= xInputToInterpolateValues.length;
        double [] yOutputInterpolatedValues = new double [numberToInterpolate];      // Array of y values, calculated for each of the input xInputToInterpolateValues values
        double dummy;                       //Dummy variables

        // Main loop. Go through and calculate interpolant at each xInputToInterpolateValues value
        for (int xToInterpolateIndex=0;xToInterpolateIndex < numberToInterpolate; xToInterpolateIndex++){
            int knotVectorIdx=-1;
            for(int xInputValueIdx=0;xInputValueIdx<this.numberOfKnots;xInputValueIdx++){
                if(this.xInputValues[xInputValueIdx]>=xInputToInterpolateValues[xToInterpolateIndex]){
                    knotVectorIdx=xInputValueIdx;
                    break;
                }
            }
            dummy = xInputToInterpolateValues[xToInterpolateIndex] - this.xInputValues[knotVectorIdx - 1];
            yOutputInterpolatedValues[xToInterpolateIndex] = this.yInputValues[knotVectorIdx - 1] + dummy*(this.dVec[knotVectorIdx - 1] + dummy*(this.c2[knotVectorIdx] + dummy*this.c3[knotVectorIdx]));
        }

        return yOutputInterpolatedValues;
    }

    /**
     * Computes the interpolation (y value) af a values (x value) for a given spline interpolation
     * (object of this class).
     * @param xInputToInterpolateValue Double value to interpolate.
     * @return Double y value corresponding to the interpolation of the given x value.
     */
    public double cubicSplineSolver( double xInputToInterpolateValue){
        assert xInputValues[0]<=xInputToInterpolateValue && xInputToInterpolateValue<= xInputValues[xInputValues.length-1]: " There are values to extrapolate. This function only interpolates values.";

        double yOutputInterpolatedValues;      // Array of y values, calculated for each of the input xInputToInterpolateValue values
        double dummy;                       //Dummy variables

        int knotVectorIdx=-1;
        for(int xInputValueIdx=0;xInputValueIdx<this.numberOfKnots;xInputValueIdx++){
            if(this.xInputValues[xInputValueIdx]>=xInputToInterpolateValue){
                knotVectorIdx=xInputValueIdx;
                    break;
            }
        }
        dummy = xInputToInterpolateValue - this.xInputValues[knotVectorIdx - 1];
        yOutputInterpolatedValues = this.yInputValues[knotVectorIdx - 1] + dummy*(this.dVec[knotVectorIdx - 1] + dummy*(this.c2[knotVectorIdx] + dummy*this.c3[knotVectorIdx]));
        return yOutputInterpolatedValues;
    }

    /**
     * Computes the interpolation (y values) af an array of values (x values). This computes all the params needed.
     * @param xInputValues X values as an array of doubles (double []). It must be ordered.
     * @param yInputValues X values as an array of doubles (double []). It must be ordered.
     * @param xInputToInterpolateValues Array if x values to interpolate (double []).
     * @return an array of y values (double []) corresponding to the interpolation of the given x values.
     */
    public static double[] cubicSplineSolver(double[] xInputValues, double[] yInputValues, double []xInputToInterpolateValues){
        assert xInputValues.length==yInputValues.length : "Sizes of knots not matching: x.length="+ xInputValues.length +" , y.length="+yInputValues.length;
        for (int i = 1; i < xInputValues.length; i++) {
            assert (xInputValues[i - 1] <= xInputValues[i]): "The xInputValues must be ordered from min to max. [..., "+ xInputValues[i - 1]+", "+ xInputValues[i]+", ...]";
        }
        for (double xInter:xInputToInterpolateValues){
            assert xInputValues[0]<=xInter && xInter<= xInputValues[xInputValues.length-1]: " There are values to extrapolate. This function only interpolates values.";
        }

        int numberOfKnots=xInputValues.length;
        int numberToInterpolate= xInputToInterpolateValues.length;
        double [] dVec = new double [numberOfKnots];            // Array of derivatives at each of the xInputValues points
        double [] yOutputInterpolatedValues = new double [numberToInterpolate];      // Array of y values, calculated for each of the input xInputToInterpolateValues values

        double [][] wk = new double[2][numberOfKnots]; // wk is a 2*M work array, used in the calculation of the derivatives at each of the entered x-values

        int nj;                                    //Array indices
        double dummy, tempx;                       //Dummy variables

        //fixed params dependent on the problem
        double[] c2 = new double [numberOfKnots],  c3 = new double [numberOfKnots], delta = new double [numberOfKnots],
                del1 = new double [numberOfKnots], del2 = new double [numberOfKnots], h = new double [numberOfKnots];

        // Differences between the x-values are stored in wk[0][. . .], starting with wk[0][1]
        // Divided differences between y-values are stored in wk[1][. . .], starting with wk[1][1]
        // Note that wk[0][0] and wk[1][0] are not filled in this loop; they are presently left unassigned
        nj = numberOfKnots - 1;
        for (int i = 0; i < nj; ++i){
            wk[0][i+1] = xInputValues[i+1] - xInputValues[i];
            wk[1][i+1] = (yInputValues[i+1] - yInputValues[i])/wk[0][i+1];
        }
        if (numberOfKnots ==1){
            return yInputValues;
        }
        if (numberOfKnots == 2){
            wk[1][0] = wk[0][0] = 1.0;
            dVec[0] = dVec[1] = wk[1][1];
            dVec[0] *= 2.0;
        }
        else {
            tempx = dummy = wk[0][1];
            wk[1][0] = wk[0][2];
            wk[0][0] = tempx + wk[0][2];
            dummy *= dummy*wk[1][2];
            dVec[0] = ((tempx + 2.0*wk[0][0])*wk[1][1]*wk[0][2] + dummy)/wk[0][0];

            for (var i = 1; i < nj; ++i){
                tempx = -(wk[0][i+1]/wk[1][i-1]);
                dVec[i] = tempx*dVec[i-1] + 3.0*(wk[0][i]*wk[1][i+1] + wk[0][i+1]*wk[1][i]);
                wk[1][i] = tempx*wk[0][i-1] + 2.0*(wk[0][i] + wk[0][i+1]);
            }

            if (numberOfKnots == 3){
                dVec[2] = 2.0*wk[1][2];
                wk[1][2] = 1.0;
                tempx = -(1.0/wk[1][1]);
            }
            else {
                tempx = wk[0][nj-1] + wk[0][nj];
                dummy = wk[0][nj]*wk[0][nj]*(yInputValues[nj-1] - yInputValues[nj-2]);
                dummy /= wk[0][nj-1];
                dVec[nj] = ((wk[0][nj] + 2.0*tempx)*wk[1][nj]*wk[0][nj-1] + dummy)/tempx;
                tempx = -(tempx/wk[1][nj-1]);
                wk[1][nj] = wk[0][nj-1];
            }
            // Complete forward pass of Gauss Elimination
            wk[1][nj] = tempx*wk[0][nj-1] + wk[1][nj];
            dVec[nj] = (tempx*dVec[nj-1] + dVec[nj])/wk[1][nj];

        }

        //Carry out back substitution
        for (var i= nj-1; i >= 0; i--){
            dVec[i] = (dVec[i] - wk[0][i]*dVec[i+1])/wk[1][i];
        }

        for ( int knotVectorIdx=1; knotVectorIdx<numberOfKnots;knotVectorIdx++){
            h [knotVectorIdx] = xInputValues[knotVectorIdx] - xInputValues[knotVectorIdx - 1];
            delta[knotVectorIdx] = (yInputValues[knotVectorIdx] - yInputValues[knotVectorIdx - 1])/h[knotVectorIdx];
            del1[knotVectorIdx]  = (dVec[knotVectorIdx - 1] - delta[knotVectorIdx])/h[knotVectorIdx];
            del2[knotVectorIdx]  = (dVec[knotVectorIdx] - delta[knotVectorIdx])/h[knotVectorIdx];

            c2[knotVectorIdx]  = -(del1[knotVectorIdx] + del1[knotVectorIdx] + del2[knotVectorIdx]);
            c3[knotVectorIdx]  = (del1[knotVectorIdx] + del2[knotVectorIdx])/h[knotVectorIdx];
        }
        // Main loop. Go through and calculate interpolant at each xInputToInterpolateValues value
        for (int xToInterpolateIndex=0;xToInterpolateIndex < numberToInterpolate; xToInterpolateIndex++){
            // Evaluate Cubic at xInputToInterpolateValues[k], j = jfirst (1) to k-1
            // =========================================================
            // Begin CHFDV
            int knotVectorIdx=-1;
            for(int xInputValueIdx=0;xInputValueIdx<numberOfKnots;xInputValueIdx++){
                if(xInputValues[xInputValueIdx]>=xInputToInterpolateValues[xToInterpolateIndex]){
                    knotVectorIdx=xInputValueIdx;
                    break;
                }
            }
            dummy = xInputToInterpolateValues[xToInterpolateIndex] - xInputValues[knotVectorIdx - 1];
            yOutputInterpolatedValues[xToInterpolateIndex] = yInputValues[knotVectorIdx - 1] + dummy*(dVec[knotVectorIdx - 1] + dummy*(c2[knotVectorIdx] + dummy*c3[knotVectorIdx]));
        }

        return yOutputInterpolatedValues;
    }

    public static void main(String[] args) {

        double [] xInput={0.5,1.,2.,3.,4.,5.};
        double [] yInput={0.25,1.,4.,9.,16.,25.};
        double [] xInputToInterpolate = {1.1,2.1,3.1,4.1, 0.51};
        double [] yOutputInterpolated = Spline.cubicSplineSolver(xInput,yInput,xInputToInterpolate);

        for(double y:yOutputInterpolated){
            System.out.println(y);
        }
        Spline spline = new Spline(xInput,yInput);
        yOutputInterpolated = spline.cubicSplineSolver(xInputToInterpolate);
        System.out.println("");
        for(double y:yOutputInterpolated){
            System.out.println(y);
        }

        System.out.println("");
        for(double x:xInputToInterpolate){
            System.out.println(spline.cubicSplineSolver(x));
        }
    }
}
