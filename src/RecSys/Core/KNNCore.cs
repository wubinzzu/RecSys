using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace RecSys.Core
{
    /// <summary>
    /// This class implements core functions for KNN-like algorithms.
    /// </summary>
    public static class KNNCore
    {
        #region GetTopKNeighborsByUser
        /// <summary>
        /// Get the top K neighbors of the target user.
        /// </summary>
        /// <param name="similaritiesByUser">The similarities matrix.</param>
        /// <param name="indexOfTargetUser">The index of target user.</param>
        /// <param name="K">The number of neighbors.</param>
        /// <returns>Each key is one user of the top K neighbors, 
        /// and the value is his similarity to the target user.</returns>
        public static Dictionary<int, double> GetTopKNeighborsByUser(Matrix<double> similaritiesByUser, int indexOfTargetUser, int K)
        {
            Debug.Assert(similaritiesByUser.RowCount == similaritiesByUser.ColumnCount, "The similarities should be in a square matrix.");
            Debug.Assert(similaritiesByUser.RowCount > K, "The total number of users is less than K + 1 ");

            Dictionary<int, double> similaritiesByTopKUser = new Dictionary<int, double>(K);

            // They will be sorted soon
            List<double> similaritiesOfNeighborsSortedBySimilarity = similaritiesByUser.Row(indexOfTargetUser).ToList();
            List<int> indexesOfNeighborsSortedBySimilarity = Enumerable.Range(0, similaritiesOfNeighborsSortedBySimilarity.Count).ToList();

            // To exclute the target user himself from neighbors
            similaritiesOfNeighborsSortedBySimilarity[indexOfTargetUser] = double.MinValue;

            // Sort the neighbors' indexes according to their similarities to the target user
            Sorting.Sort<double, int>(similaritiesOfNeighborsSortedBySimilarity, indexesOfNeighborsSortedBySimilarity);

            // Make it descending order by similarity
            similaritiesOfNeighborsSortedBySimilarity.Reverse();
            indexesOfNeighborsSortedBySimilarity.Reverse();

            for (int i = 0; i < K; ++i)
            {
                // Be very careful about the index
                // indexesOfNeighborsSortedBySimilarity[0] will give 
                // the index (in similaritiesByUser) of the most similar neighbor
                // and i is the index in the sorted similaritiesOfNeighbors 
                similaritiesByTopKUser[indexesOfNeighborsSortedBySimilarity[i]] = similaritiesOfNeighborsSortedBySimilarity[i];
            }

            return similaritiesByTopKUser;
        }
        #endregion

        #region Obsolete
        [Obsolete("Don't use it, its old implementation of GetTopK)")]
        public static Dictionary<int, double> GetTopK2(DenseMatrix userSimilarities, int uidTarget, int K)
        {
            Dictionary<int, double> topK = new Dictionary<int, double>(K);
            Vector<double> uidSimilarities = userSimilarities.Row(uidTarget);

            double minSimilarity = double.MinValue;
            int minUid = int.MinValue;
            foreach (Tuple<int, double> entry in uidSimilarities.EnumerateIndexed(Zeros.AllowSkip))
            {
                int uid = entry.Item1;
                double similarity = entry.Item2;
                if (uid == uidTarget) { continue; } // A user is not a neighbor of himself
                if (topK.Count < K)  // Fill the top K list untill it is full
                {
                    topK[uid] = similarity;
                    if (topK.Count == K)
                    {
                        // Find the least similar neighbor when it is full
                        minUid = topK.Aggregate((l, r) => l.Value < r.Value ? l : r).Key;
                        minSimilarity = topK[minUid];
                    }
                }
                else if (similarity > minSimilarity)
                {
                    // Replace the least similar neighbor
                    topK.Remove(minUid);    // The first time it does nothing as the minUid is not set
                    topK[uid] = similarity;

                    // Find the least similar neighbor
                    minUid = topK.Aggregate((l, r) => l.Value < r.Value ? l : r).Key;
                    minSimilarity = topK[minUid];
                }
            }

            return topK;
        }
        #endregion
    }
}
