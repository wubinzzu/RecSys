using RecSys.Numerical;

namespace RecSys.Evaluation
{
    public class MAE
    {
        public static double Evaluate(RatingMatrix correctRatingMatrix, RatingMatrix predictedRatingMatrix)
        {
            return (correctRatingMatrix.Matrix - predictedRatingMatrix.Matrix)
                .ColumnAbsoluteSums().Sum() / correctRatingMatrix.NonZerosCount;
        }
    }
}
