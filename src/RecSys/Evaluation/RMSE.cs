using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Numerical;
using System;
using System.Diagnostics;

namespace RecSys.Evaluation
{
    /// <summary>
    /// Root-Mean-Square Error.
    /// See https://www.kaggle.com/wiki/RootMeanSquaredError
    /// </summary>
    public class RMSE
    {
        public static double Evaluate(RatingMatrix correctMatrix, RatingMatrix predictedMatrix)
        {
            Debug.Assert(correctMatrix.NonZerosCount == predictedMatrix.NonZerosCount);
            double enumerator = (predictedMatrix.Matrix - correctMatrix.Matrix).FrobeniusNorm();
            return enumerator / Math.Sqrt(correctMatrix.NonZerosCount);
        }
    }
}
