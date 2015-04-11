using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Numerical;
using RecSys.Ordinal;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Core
{
    public static class Metric
    {
        public enum SimilarityMetric { PearsonPreferenceRelation, PearsonRating, CosinePreferenceRelation, CosineRating };

        #region Preference Relation Pearson
        private static double PearsonPR()
        {
            throw new NotImplementedException();
        }
        #endregion

        #region Rating Pearson
        public static double PearsonR(RatingMatrix R, int a, int b)
        {
            // TODO: I'm wondering when we compute similarity,
            // should we count zeros in the vectors or not?
            // Why it is not Distance-1????!!
            return 1 - Distance.Pearson(R.GetRow(a), R.GetRow(b));
        }
        #endregion

        #region Preference Relation Cosine
        public static double cosinePR(PreferenceRelations PR, int u_a, int u_b)
        {
            SparseMatrix pr_a = PR[u_a];
            SparseMatrix pr_b = PR[u_b];

            int agreedCount = 0;	// The number of preference relations agreed between users a and b

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

            // Because we avoided double counting, so we divid each count by 2
            Debug.Assert((pr_a.NonZerosCount).IsEven());  // Because pr's upper triangular should be a mirror of the lower triangular

            // Large number, be careful about overflow
            double normalization = checked(Math.Sqrt((double)pr_a.NonZerosCount / 2) * Math.Sqrt((double)pr_b.NonZerosCount / 2));

            // TODO: Extreme small, is it correct?
            return agreedCount / normalization;
        }
        #endregion

        #region Rating Cosine
        public static double CosineR(RatingMatrix R, int a, int b)
        {
            // TODO: I'm wondering when we compute similarity,
            // should we count zeros in the vectors or not?
            return Distance.Cosine(R.GetRow(a).ToArray(), R.GetRow(b).ToArray());
        }
        #endregion

    }
}
