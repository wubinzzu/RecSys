using RecSys.Numerical;
using System;
using System.Diagnostics;

namespace RecSys.Evaluation
{
    public class RMSE
    {
        public static double Evaluate(RatingMatrix correctRatingMatrix, RatingMatrix predictedRatingMatrix)
        {
            Debug.Assert(correctRatingMatrix.NonZerosCount == predictedRatingMatrix.NonZerosCount);
            double enumerator = (predictedRatingMatrix.Matrix - correctRatingMatrix.Matrix).FrobeniusNorm();
            return enumerator / Math.Sqrt(correctRatingMatrix.Matrix.NonZerosCount);
        }
    }
}
