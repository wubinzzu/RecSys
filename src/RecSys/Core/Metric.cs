using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Statistics;
using RecSys.Numerical;
using RecSys.Ordinal;
using System;
using System.Collections.Generic;
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
        public static void GetPearsonOfRows(RatingMatrix R, int maxCountOfNeighbors,
            double strongSimilarityThreshold, out SimilarityData neighborsByObject,
            out HashSet<Tuple<int, int>> strongSimilarityIndicators)
        {
            ComputeSimilarities(R.Matrix, SimilarityMetric.PearsonRating, maxCountOfNeighbors,
                strongSimilarityThreshold, out neighborsByObject, out strongSimilarityIndicators);
        }
        public static void GetCosineOfRows(RatingMatrix R, int maxCountOfNeighbors, 
            double strongSimilarityThreshold, out SimilarityData neighborsByObject)
        {
            HashSet<Tuple<int, int>> foo;
            ComputeSimilarities(R.Matrix, SimilarityMetric.CosineRating, maxCountOfNeighbors,
                strongSimilarityThreshold, out neighborsByObject, out foo);
        }
        public static void GetPearsonOfColumns(RatingMatrix R, int maxCountOfNeighbors,
            double strongSimilarityThreshold, out SimilarityData neighborsByObject,
            out HashSet<Tuple<int, int>> strongSimilarityIndicators)
        {
            ComputeSimilarities(R.Matrix.Transpose(), SimilarityMetric.PearsonRating, maxCountOfNeighbors,
                strongSimilarityThreshold, out neighborsByObject, out strongSimilarityIndicators);
        }
        public static void GetCosineOfColumns(RatingMatrix R, int maxCountOfNeighbors,
            double strongSimilarityThreshold, out SimilarityData neighborsByObject,
            out HashSet<Tuple<int, int>> strongSimilarityIndicators)
        {
            // Just rotate the matrix
            ComputeSimilarities(R.Matrix.Transpose(), SimilarityMetric.CosineRating, maxCountOfNeighbors,
                strongSimilarityThreshold, out neighborsByObject, out strongSimilarityIndicators);
        }
        public static void GetCosineOfPrefRelations(PrefRelations PR, int maxCountOfNeighbors,
                        double strongSimilarityThreshold, out SimilarityData neighborsByObject)
        {
            HashSet<Tuple<int, int>> foo;
            ComputeSimilarities(PR, SimilarityMetric.CosinePrefRelations, maxCountOfNeighbors,
    strongSimilarityThreshold, out neighborsByObject, out foo);
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
        private static Matrix<double> ComputeSimilarities(Matrix<double> R, Metric.SimilarityMetric similarityMetric)
        {
            int dimension = R.RowCount;

            // For all metrics we use (Pearson and Cosine) the max similarity is 1.0
            Matrix<double> similarities = Matrix.Build.DenseDiagonal(dimension, 1);

            /*
            // Compute similarity for the lower triangular
            Object lockMe = new Object();
            int count = 0;
            Parallel.ForEach(R.EnumerateRowsIndexed(), row_a =>
            {
                int indexOfRow_a = row_a.Item1;
                Vector<double> valuesOfRow_a = row_a.Item2;
                Vector<double> similaritiesOfRow_a = Vector.Build.Dense(dimension);

                foreach (var row_b in R.EnumerateRowsIndexed(indexOfRow_a+1,R.RowCount-indexOfRow_a-1))
                {
                    int indexOfRow_b = row_b.Item1;
                    Vector<double> valuesOfRow_b = row_b.Item2;

                    // Compute only the upper triangular
                    //if (indexOfRow_a > indexOfRow_b)
                    //{
                        switch (similarityMetric)
                        {
                            case Metric.SimilarityMetric.CosineRating:
                                similaritiesOfRow_a[indexOfRow_b] = CosineR(valuesOfRow_a, valuesOfRow_b);
                                break;
                            case Metric.SimilarityMetric.PearsonRating:
                                similaritiesOfRow_a[indexOfRow_b] = PearsonR(valuesOfRow_a, valuesOfRow_b);
                                break;
                        }
                    //}
                }

                lock (lockMe)
                {
                    similarities.SetRow(indexOfRow_a, similaritiesOfRow_a);
                    count++;
                    Utils.PrintEpoch("Progress current/total", count, R.RowCount);
                }
            });
            */

            // Compute similarity for the lower triangular
            #region Old implementation
            
            Object lockMe = new Object();
            Parallel.For(0, dimension, i =>
            {
                Utils.PrintEpoch("Progress current/total", i, dimension);

                for (int j = 0; j < dimension; j++)
                {
                    if (i == j) { continue; }// Skip the diagonal
                    else if (i > j)
                    {
                        switch (similarityMetric)
                        {
                            case Metric.SimilarityMetric.CosineRating:
                                double cosine = Metric.CosineR(R.Row(i), R.Row(j));
                                lock (lockMe)
                                {
                                    similarities[i, j] = cosine;
                                }
                                break;
                            case Metric.SimilarityMetric.PearsonRating:
                                double pearson = Metric.PearsonR(R.Row(i), R.Row(j));
                                lock (lockMe)
                                {
                                    similarities[i, j] = pearson;
                                }
                                break;
                        }
                    }
                }
            });
            #endregion

            // Copy similarity values from upper triangular to lower triangular
            similarities = similarities + similarities.Transpose();

            return similarities;
        }
        #endregion

        #region Compute topK neighbors of each row
        private static void ComputeSimilarities(Matrix<double> R, 
            Metric.SimilarityMetric similarityMetric, int maxCountOfNeighbors,
            double minSimilarityThreshold,  out SimilarityData neighborsByObject,
            out HashSet<Tuple<int, int>> strongSimilarityIndicators)
        {
            int dimension = R.RowCount;
            List<Vector<double>> rows = new List<Vector<double>>(R.EnumerateRows());

            // I assume that the rows are enumerated from first to last
            Debug.Assert(rows[0].Sum() == R.Row(0).Sum());
            Debug.Assert(rows[rows.Count - 1].Sum() == R.Row(rows.Count - 1).Sum());

            List<Tuple<int, int>> strongSimilarityIndicators_out = new List<Tuple<int, int>>();

            SimilarityData neighborsByObject_out = new SimilarityData(maxCountOfNeighbors);

            Object lockMe = new Object();
            Parallel.For(0, dimension, indexOfRow =>
            {
                Utils.PrintEpoch("Progress current/total", indexOfRow, dimension);
                Dictionary<Tuple<int, int>,double> similarityCache = new Dictionary<Tuple<int, int>,double>();
                List<Tuple<int, int>> strongSimilarityIndocatorCache = new List<Tuple<int, int>>();

                for (int indexOfNeighbor = 0; indexOfNeighbor < dimension; indexOfNeighbor++)
                {
                    if (indexOfRow == indexOfNeighbor) { continue; } // Skip self similarity

                    else if (indexOfRow > indexOfNeighbor)
                    {
                        switch (similarityMetric)
                        {
                            case Metric.SimilarityMetric.CosineRating:
                                double cosine = Metric.CosineR(rows[indexOfRow],rows[indexOfNeighbor]);
                                    if(cosine >  minSimilarityThreshold)
                                    {
                                        strongSimilarityIndocatorCache.Add(new Tuple<int, int>(indexOfRow, indexOfNeighbor));
                                    }
                                    similarityCache[new Tuple<int, int>(indexOfRow, indexOfNeighbor)] = cosine;

                                break;
                            case Metric.SimilarityMetric.PearsonRating:
                                double pearson = Metric.PearsonR(rows[indexOfRow], rows[indexOfNeighbor]);
                                    if (pearson> minSimilarityThreshold)
                                    {
                                        strongSimilarityIndocatorCache.Add(new Tuple<int, int>(indexOfRow, indexOfNeighbor));
                                    }
                                    similarityCache[new Tuple<int, int>(indexOfRow, indexOfNeighbor)] = pearson;

                                break;
                        }
                    }
                }

                lock (lockMe)
                {
                    foreach(var entry in similarityCache)
                    {
                        neighborsByObject_out.AddSimilarityData(entry.Key.Item1, entry.Key.Item2, entry.Value);
                        neighborsByObject_out.AddSimilarityData(entry.Key.Item2, entry.Key.Item1, entry.Value);
                    }
                    strongSimilarityIndicators_out.AddRange(strongSimilarityIndocatorCache);
                }
            });

            neighborsByObject = neighborsByObject_out;
            neighborsByObject.SortAndRemoveNeighbors();
            strongSimilarityIndicators = new HashSet<Tuple<int,int>>(strongSimilarityIndicators_out);
        }
        #endregion


        #region All preference relations based metrics share the same computation flow in this function
        /// <summary>
        /// Switch between different metrics.
        /// </summary>
        /// <param name="PR"></param>
        /// <param name="similarityMetric"></param>
        /// <returns></returns>
        private static void ComputeSimilarities(PrefRelations PR,
            Metric.SimilarityMetric similarityMetric, int maxCountOfNeighbors,
                        double minSimilarityThreshold, out SimilarityData neighborsByObject,
            out HashSet<Tuple<int, int>> strongSimilarityIndicators)
        {
            int dimension = PR.UserCount;
            HashSet<Tuple<int, int>> strongSimilarityIndicators_out = new HashSet<Tuple<int, int>>();
            SimilarityData neighborsByObject_out = new SimilarityData(maxCountOfNeighbors);

            // Compute similarity for the lower triangular
            Object lockMe = new Object();
            Parallel.For(0, dimension, i =>
            {
                Utils.PrintEpoch("Progress current/total", i, dimension);

                for (int j = 0; j < dimension; j++)
                {
                    if (i == j) { continue; } // Skip self similarity

                    else if (i > j)
                    {
                        switch (similarityMetric)
                        {
                            case SimilarityMetric.CosinePrefRelations:
                                double cosinePR = Metric.cosinePR(PR, i, j);
                                lock (lockMe)
                                {
                                    if (cosinePR > minSimilarityThreshold)
                                    {
                                        strongSimilarityIndicators_out.Add(new Tuple<int, int>(i, j));
                                    }
                                    neighborsByObject_out.AddSimilarityData(i, j, cosinePR);
                                    neighborsByObject_out.AddSimilarityData(j, i, cosinePR);
                                }
                                break;
                            // More metrics to be added here.
                        }
                    }
                }
            });

            neighborsByObject = neighborsByObject_out;
            strongSimilarityIndicators = strongSimilarityIndicators_out;
        }
        #endregion

        #region Rating Pearson
        private static double PearsonR(Vector<double> Vector_a, Vector<double> Vector_b)
        {
            double correlation = Correlation.Pearson(Vector_a,Vector_b);
            if (double.IsNaN(correlation))
            {
                // This means one of the row has 0 standard divation,
                // it does not correlate to anyone
                // so I assign the correlatino to be 0. however, strictly speaking, it should be left NaN
                correlation = 0;
            }
            return correlation;
        }
        #endregion

        #region Rating Cosine
        private static double CosineR(Vector<double> Vector_a, Vector<double> Vector_b)
        {
            return Vector_a.DotProduct(Vector_b) / (Vector_a.L2Norm() * Vector_b.L2Norm());
            //return Distance.Cosine(R.Row(a).ToArray(), R.Row(b).ToArray());
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
