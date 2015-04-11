using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Numerical;
using RecSys.Ordinal;
using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace RecSys.Core
{
    /// <summary>
    /// This class implements different similarity metrics.
    /// </summary>
    public static class Metric
    {
        #region Public interfaces to compute similarities of matrix/preference relations
        public static DenseMatrix GetPearsonOfRows(RatingMatrix R)
        {
            return ComputeSimilarities(R, SimilarityMetric.PearsonRating);
        }
        public static DenseMatrix GetCosineOfRows(RatingMatrix R)
        {
            throw new NotImplementedException();
        }
        public static DenseMatrix GetPearsonOfColumns(RatingMatrix R)
        {
            // Just rotate the matrix
            throw new NotImplementedException();
        }
        public static DenseMatrix GetCosineOfColumns(RatingMatrix R)
        {
            // Just rotate the matrix
            throw new NotImplementedException();
        }
        public static DenseMatrix GetCosineOfPrefRelations(PrefRelations PR)
        {
            return ComputeSimilarities(PR, SimilarityMetric.CosinePrefRelations);
        }

        #endregion

        #region Private implementations
        private enum SimilarityMetric { PearsonPrefRelations, PearsonRating, CosinePrefRelations, CosineRating };

        #region All rating-based metrics share the same computation flow in this function
        /// <summary>
        /// Switch between different metrics.
        /// </summary>
        /// <param name="R"></param>
        /// <param name="similarityMetric"></param>
        /// <returns></returns>
        private static DenseMatrix ComputeSimilarities(RatingMatrix R, Metric.SimilarityMetric similarityMetric)
        {
            int dimension = R.UserCount;

            // For all metrics we use (Pearson and Cosine) the max similarity is 1.0
            DenseMatrix similarities = DenseMatrix.OfMatrix(DenseMatrix.Build.DenseDiagonal(dimension, 1));

            // Compute similarity for the lower triangular
            Object lockMe = new Object();
            Parallel.For(0, dimension, i =>
            {
                Utils.PrintEpoch("Progress user/total", i, dimension);

                for (int j = 0; j < dimension; j++)
                {
                    if (i == j) { continue; }// Skip the diagonal
                    else if (i > j)
                    {
                        switch (similarityMetric)
                        {
                            case Metric.SimilarityMetric.CosineRating:
                                double cosine = Metric.CosineR(R, i, j);
                                lock (lockMe)
                                {
                                    similarities[i, j] = cosine;
                                }
                                break;
                            case Metric.SimilarityMetric.PearsonRating:
                                double pearson = Metric.PearsonR(R, i, j);
                                lock (lockMe)
                                {
                                    similarities[i, j] = pearson;
                                }
                                break;
                        }
                    }
                }
            });

            // Copy similarity values from lower triangular to upper triangular
            similarities = DenseMatrix.OfMatrix(similarities + similarities.Transpose()
                - DenseMatrix.CreateIdentity(similarities.RowCount));

            return similarities;
        }
        #endregion

        #region All preference relations based metrics share the same computation flow in this function
        /// <summary>
        /// Switch between different metrics.
        /// </summary>
        /// <param name="PR"></param>
        /// <param name="similarityMetric"></param>
        /// <returns></returns>
        private static DenseMatrix ComputeSimilarities(PrefRelations PR, SimilarityMetric similarityMetric)
        {
            int dimension = PR.UserCount;

            // For all metrics we use (Pearson and Cosine) the max similarity is 1.0
            DenseMatrix similarities = DenseMatrix.OfMatrix(DenseMatrix.Build.DenseDiagonal(dimension, 1));

            // Compute similarity for the lower triangular
            Object lockMe = new Object();
            Parallel.For(0, dimension, i =>
            {
                Utils.PrintEpoch("Get similarity user/total", i, dimension);

                for (int j = 0; j < dimension; j++)
                {
                    if (i == j) { continue; }// Skip the diagonal
                    else if (i > j)
                    {
                        switch (similarityMetric)
                        {
                            case SimilarityMetric.CosinePrefRelations:
                                double cosinePR = Metric.cosinePR(PR, i, j);
                                lock (lockMe)
                                {
                                    similarities[i, j] = cosinePR;
                                }
                                break;
                            // More metrics to be added here.
                        }
                    }
                }
            });
            // Copy similarity values from lower triangular to upper triangular
            similarities = DenseMatrix.OfMatrix(similarities + similarities.Transpose()
                - DenseMatrix.CreateIdentity(similarities.RowCount));

            Debug.Assert(similarities[0, 0] == 1, "The similarities[0,0] should be 1 for Pearson correlation.");

            return similarities;
        }
        #endregion

        #region Rating Pearson
        private static double PearsonR(RatingMatrix R, int a, int b)
        {
            // TODO: I'm wondering when we compute similarity,
            // should we count zeros in the vectors or not?
            // Why it is not Distance-1????!!
            return 1 - Distance.Pearson(R.GetRow(a), R.GetRow(b));
            //return 1 - Distance.Pearson(R.Matrix.Row(a), R.Matrix.Row(b));
        }
        #endregion

        #region Rating Cosine
        private static double CosineR(RatingMatrix R, int a, int b)
        {
            // TODO: I'm wondering when we compute similarity,
            // should we count zeros in the vectors or not?
            return Distance.Cosine(R.GetRow(a).ToArray(), R.GetRow(b).ToArray());
        }
        #endregion

        #region Preference Relation Pearson
        private static double PearsonPR()
        {
            throw new NotImplementedException();
        }
        #endregion

        #region Preference Relation Cosine
        private static double cosinePR(PrefRelations PR, int u_a, int u_b)
        {
            SparseMatrix pr_a = PR[u_a];
            SparseMatrix pr_b = PR[u_b];

            Debug.Assert(pr_a.Trace() == SparseMatrix.Zero, "The diagonal of user preference relation matrix should be left empty.");
            Debug.Assert(pr_b.Trace() == SparseMatrix.Zero, "The diagonal of user preference relation matrix should be left empty.");

            // The number of preference relations agreed between users a and b
            int agreedCount = pr_a.Fold2((count, prefOfA, prefOfB) =>
                    count + (prefOfA == prefOfB ? 1 : 0), 0, pr_b, Zeros.AllowSkip);

            #region Obsolate naive implementation
            /*
            // TODO: there should be a faster lambda way of doing this 
            // Loop through all non-zero elements
            foreach (Tuple<int, int, double> element in pr_a.EnumerateIndexed(Zeros.AllowSkip))
            {
                int item_i = element.Item1;
                int item_j = element.Item2;
                double preference_a = element.Item3;
                // Because pr_ij is just the reverse of pr_ji,
                // we count only i-j to avoid double counting
                // and also reduce the number of calling pr_b[]
                if (item_i > item_j)
                {
                    if (preference_a == pr_b[item_i, item_j])
                    {
                        ++agreedCount;
                    }
                }
            }
            */
            #endregion

            // Multiplicaiton result can be too large and cause overflow,
            // therefore we do Sqrt() first and then multiply
            double normalization = checked(Math.Sqrt((double)pr_a.NonZerosCount) * Math.Sqrt((double)pr_b.NonZerosCount));

            // Very small value
            return agreedCount / normalization;
        }
        #endregion

        #endregion
    }
}
