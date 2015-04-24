using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Data.Text;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using RecSys.Core;

namespace RecSys.Numerical
{
    public static class UserKNN
    {

        #region UserKNN
        /// <summary>
        /// The user-based KNN collaborative filtering described in paper: 
        /// Resnick, P., et al., "GroupLens: an open architecture for collaborative filtering of netnews", 1994.
        /// Link: http://dx.doi.org/10.1145/192844.192905
        /// </summary>
        /// <param name="R_train"></param>
        /// <param name="R_unknown"></param>
        /// <param name="K"></param>
        /// <returns></returns>
        public static DataMatrix PredictRatings(DataMatrix R_train, DataMatrix R_unknown, SimilarityData neighborsByUser, int K)
        {
            // Debug
            Debug.Assert(R_train.UserCount == R_unknown.UserCount);
            Debug.Assert(R_train.ItemCount == R_unknown.ItemCount);
            int cappedCount = 0, globalMeanCount = 0;

            // This matrix stores predictions
            DataMatrix R_predicted = new DataMatrix(R_unknown.UserCount, R_unknown.ItemCount);

            // Basic statistics from train set
            double globalMean = R_train.GetGlobalMean();
            Vector<double> meanByUser = R_train.GetUserMeans();
            Vector<double> meanByItem = R_train.GetItemMeans();

            // Predict ratings for each test user
            // Single thread appears to be very fast, parallel.foreach is unnecessary
            Object lockMe = new Object();
            Parallel.ForEach(R_unknown.Users, user =>
            {
                int indexOfUser = user.Item1;
                RatingVector userRatings = new RatingVector(R_train.GetRow(indexOfUser));
                RatingVector unknownRatings = new RatingVector(user.Item2);

                Utils.PrintEpoch("Predicting user/total", indexOfUser, R_train.UserCount);

                // Note that there are more than K neighbors in the list (sorted by similarity)
                // we will use the top-K neighbors WHO HAVE RATED THE ITEM
                // For example we have 200 top neighbors, and we hope there are
                // K neighbors in the list have rated the item. We can't keep
                // everyone in the neighbor list because there are too many for large data sets
                var topNeighborsOfUser = neighborsByUser[indexOfUser];
                //Dictionary<int, double> topKNeighbors = KNNCore.GetTopKNeighborsByUser(userSimilarities, indexOfUser, K);

                double meanOfUser = meanByUser[indexOfUser];

                // Loop through each ratingto be predicted
                foreach (Tuple<int, double> unknownRating in unknownRatings.Ratings)
                {
                    int itemIndex = unknownRating.Item1;
                    double prediction;

                    // TODO: we actually should use the Top-K neighbors
                    // that have rated this item, otherwise we may have
                    // only a few neighbors rated this item

                    // Compute the average rating on item iid given 
                    // by the top K neighbors. Each rating is offsetted by
                    // the neighbor's average and weighted by the similarity
                    double weightedSum = 0;
                    double weightSum = 0;
                    int currentTopKCount = 0;
                    foreach (KeyValuePair<int, double> neighbor in topNeighborsOfUser)
                    {
                        int neighborIndex = neighbor.Key;
                        double similarityOfNeighbor = neighbor.Value;
                        double itemRatingOfNeighbor = R_train[neighborIndex, itemIndex];

                        // We count only if the neighbor has seen this item before
                        if (itemRatingOfNeighbor != 0)
                        {
                            weightSum += similarityOfNeighbor;
                            weightedSum += (itemRatingOfNeighbor - meanByUser[neighborIndex]) * similarityOfNeighbor;
                            currentTopKCount++;
                            if (currentTopKCount >= K) { break; }   // Stop when we have seen K neighbors
                        }
                    }
                    // A zero weightedSum means this is a cold item and global mean will be assigned by default
                    if (weightedSum != 0)
                    {
                        prediction = meanOfUser + weightedSum / weightSum;
                    }
                    else
                    {
                        prediction = globalMean;
                        globalMeanCount++;
                    }

                    // Cap the ratings
                    if (prediction > Config.Ratings.MaxRating)
                    {
                        cappedCount++;
                        prediction = Config.Ratings.MaxRating;
                    }
                    if (prediction < Config.Ratings.MinRating)
                    {
                        cappedCount++;
                        prediction = Config.Ratings.MinRating;
                    }

                    lock (lockMe)
                    {
                        R_predicted[indexOfUser, itemIndex] = prediction;
                    }
                }
            });
            Utils.PrintValue("# capped predictions", cappedCount.ToString("D"));
            Utils.PrintValue("# default predictions", globalMeanCount.ToString("D"));
            return R_predicted;
        }
        #endregion

    }
}
