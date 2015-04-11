using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RecSys.Core;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;
using RecSys.Numerical;

namespace RecSys.Ordinal
{
    /// <summary>
    /// Implementation of the PrefUserKNN algorithm described in:
    /// Brun, A., Hamad, A., Buffet, O., & Boyer, A. (2010). 
    /// Towards preference relations in recommender systems. Workshop in ECML-PKDD.
    /// </summary>
    public static class PrefUserKNN
    {
        public static RatingMatrix PredictRatings(PrefRelations PR_train,
            RatingMatrix R_unknown, int K)
        {
            Debug.Assert(PR_train.UserCount == R_unknown.UserCount);
            Debug.Assert(PR_train.ItemCount == R_unknown.ItemCount);

            // This matrix stores predictions
            RatingMatrix R_predicted = new RatingMatrix(R_unknown.UserCount, R_unknown.ItemCount);

            // This can be considered as the R_train in standard UserKNN
            SparseMatrix positionMatrix = PR_train.GetPositionMatrix();
            RatingMatrix ratingMatrixFromPositions = new RatingMatrix(positionMatrix);

            Vector<double> meanByUser = ratingMatrixFromPositions.GetUserMeans();
            Vector<double> meanByItem = ratingMatrixFromPositions.GetItemMeans();
            double globalMean = ratingMatrixFromPositions.GetGlobalMean();

            // Predict positions for each test user
            // Appears to be very fast, parallel.foreach is unnecessary
            foreach (Tuple<int, Vector<double>> user in R_unknown.Users)
            {
                int indexOfUser = user.Item1;
                Vector<double> indexesOfUnknownRatings = user.Item2;

                Utils.PrintEpoch("Predicting user/total", indexOfUser, PR_train.UserCount);

                Dictionary<int, double> topKNeighbors = KNNCore.GetTopKNeighborsByUser(PR_train.UserSimilarities, indexOfUser, K);

                double meanOfUser = meanByUser[indexOfUser];

                // Loop through each position to be predicted
                foreach (Tuple<int, double> unknownRating in indexesOfUnknownRatings.EnumerateIndexed(Zeros.AllowSkip))
                {
                    int indexOfUnknownItem = unknownRating.Item1;

                    // Compute the position of this item for the user
                    // by combining neighbors' positions on this item
                    double weightedSum = 0;
                    double weightSum = 0;
                    int itemSeenCount = 0;
                    foreach (KeyValuePair<int, double> neighbor in topKNeighbors)
                    {
                        int indexOfNeighbor = neighbor.Key;
                        double similarityOfNeighbor = neighbor.Value;
                        double itemPositionOfNeighbor = ratingMatrixFromPositions[indexOfNeighbor, indexOfUnknownItem];

                        // We count only if the neighbor has seen this item before
                        if (itemPositionOfNeighbor != 0)
                        {
                            // Recall that we use a small constant to hold 0
                            // we revert it back here
                            if (itemPositionOfNeighbor == Config.ZeroInSparseMatrix)
                            {
                                itemPositionOfNeighbor = 0;
                            }
                            weightSum += similarityOfNeighbor;
                            weightedSum += (itemPositionOfNeighbor - meanByUser[indexOfNeighbor]) * similarityOfNeighbor;
                            itemSeenCount++;
                        }
                    }

                    // If any neighbor has seen this item
                    if (itemSeenCount != 0)
                    {
                        // TODO: Add user mean may improve the performance
                        R_predicted[indexOfUser, indexOfUnknownItem] = meanOfUser + weightedSum / weightSum;
                    }
                    else
                    {
                        R_predicted[indexOfUser, indexOfUnknownItem] = globalMean;
                    }
                }
            }//);
            return R_predicted;
        }

        #region PrefUserKNN
        public static Dictionary<int, List<int>> RecommendTopN(PrefRelations PR_train, int K, List<int> targetUsers, int topN)
        {
            Dictionary<int, List<int>> topNItemsByUser = new Dictionary<int, List<int>>(targetUsers.Count);

            int userCount = PR_train.UserCount;
            int itemCount = PR_train.ItemCount;
            SparseMatrix positionMatrix = PR_train.GetPositionMatrix();

            // Make recommendations to each target user
            foreach (int indexOfUser in targetUsers)
            {
                Utils.PrintEpoch("Current user/total", indexOfUser, targetUsers.Count);

                // TODO: should have a default list of popular items in case of cold users
                Dictionary<int, double> topNItems = new Dictionary<int, double>(topN);   // To store recommendations for indexOfUser
                Dictionary<int, double> topKNeighbors = KNNCore.GetTopKNeighborsByUser(PR_train.UserSimilarities, indexOfUser, K);
                SparseVector predictedPositionsOfUser = new SparseVector(itemCount);

                // Compute the predicted position of each item for indexOfUser
                for (int indexOfItem = 0; indexOfItem < itemCount; ++indexOfItem)
                {
                    // Compute the position of this item for the user
                    // by combining neighbors' positions on this item
                    double weightedSum = 0;
                    double weightSum = 0;
                    int itemSeenCount = 0;
                    foreach (KeyValuePair<int, double> neighbor in topKNeighbors)
                    {
                        int indexOfNeighbor = neighbor.Key;
                        double similarityOfNeighbor = neighbor.Value;
                        double itemPositionOfNeighbor = positionMatrix[indexOfNeighbor, indexOfItem];

                        // TODO: Zero means it is not seen by the neighbor but 
                        // it may also be the position value of 0
                        if (itemPositionOfNeighbor != 0)
                        {
                            weightSum += similarityOfNeighbor;
                            weightedSum += itemPositionOfNeighbor * similarityOfNeighbor;
                            itemSeenCount++;
                        }
                    }

                    // If any neighbor has seen this item
                    if (itemSeenCount != 0)
                    {
                        // TODO: Add user mean may improve the performance
                        predictedPositionsOfUser[indexOfItem] = weightedSum / weightSum;
                    }
                }
                List<int> indexesOfItemSortedByPosition = Enumerable.Range(0, itemCount).ToList();

                Sorting.Sort(predictedPositionsOfUser, indexesOfItemSortedByPosition);
                indexesOfItemSortedByPosition.Reverse(); // Make it descending order by position
                // Add the top N items for user uid
                topNItemsByUser[indexOfUser] = indexesOfItemSortedByPosition.GetRange(0, topN);
            }

            return topNItemsByUser;
            #region Old version
            /*
            //===============Initialize variables==================

            // Recommendations are stored here indexed by user id
            Dictionary<int, List<int>> userRecommendations = new Dictionary<int, List<int>>(targetUsers.Count);

            int userCount = PR_train.UserCount;
            int itemCount = PR_train.ItemCount;

            // Build the item position matrix
            // each element indicates the position(kind of goodness) of an item to the user
            SparseMatrix itemPositions = new SparseMatrix(userCount, itemCount);

            Object lockMe = new Object();
            Parallel.ForEach(PR_train.GetAllPreferenceRelations, pair =>
            {
                int uid = pair.Key;
                Utilities.PrintEpoch("Current user/total", uid, userCount);
                SparseMatrix userPreferences = pair.Value;
                foreach (Tuple<int, Vector<double>> preferences in userPreferences.EnumerateRowsIndexed())
                {
                    int iid = preferences.Item1;
                    SparseVector iidPreferences = SparseVector.OfVector(preferences.Item2);
                    // The number of items that are preferred to item iid
                    int preferredCount = 0;
                    // The number of items that are less preferred to item iid
                    int lessPreferredCount = 0;
                    // The number of items (other than item iid) that are equally preferred to item iid
                    // TODO: I'm not sure if we should count unknown preferences or not?
                    int equallyPreferredCount = 0;

                    // Note: don't use the Count() method it won't skip Zeros
                    foreach (double preference in iidPreferences.Enumerate(Zeros.AllowSkip))
                    {
                        if (preference == Config.Preferences.Preferred)
                            ++preferredCount;
                        else if (preference == Config.Preferences.LessPreferred)
                            ++lessPreferredCount;
                        else if (preference == Config.Preferences.EquallyPreferred)
                            ++equallyPreferredCount;
                        else { Debug.Assert(false, "We should not see any non-match value here."); }
                    }

                    double position = ((double)lessPreferredCount - preferredCount) / (preferredCount + lessPreferredCount + equallyPreferredCount);

                    Debug.Assert(position >= -1 && position <= 1);  // According to the paper
                    if (position == 0) { Debug.Assert(preferredCount == lessPreferredCount); }  // According to the paper

                    lock (lockMe)
                    {
                        itemPositions[uid, iid] = position;
                    }
                }
            });

            // Need to cache the items appeared in each user's profile
            // as we won't consider unseen items as recommendations
            Dictionary<int, List<int>> seenItemsByUser = PR_train.GetSeenItemsByUser();

            Matrix positionMatrix = PR_train.GetPositionMatrix();

            Console.WriteLine("Recommending user/total");

            // Make recommendations for each target user
            foreach (int uid in targetUsers)
            {

                Utilities.PrintEpoch("Current user/total", uid, targetUsers.Count);

                // TODO: should have a default list of popular items in case of cold users
                Dictionary<int, double> topN = new Dictionary<int, double>(topNCount);   // To store recommendations for user uid

                Dictionary<int, double> topK = KNNCore.GetTopK(PR_train.UserSimilarities, uid, K);

                // Get a list of all candidate items
                List<int> candidateItems = new List<int>();
                foreach (int uid_neighbor in topK.Keys)
                {
                    // TODO: union will remove duplicates, seems to be expensive here
                    candidateItems = candidateItems.Union(seenItemsByUser[uid_neighbor]).ToList();
                }

                // Loop through all candidate items
                double minPosition = double.MinValue;
                int min_iid = int.MinValue;
                foreach (int iid in candidateItems)
                {
                    // Compute the average position on item iid given 
                    // by the top K neighbors. Each position is weighted 
                    // by the similarity to the target user
                    double weightedSum = 0;
                    double weightSum = 0;
                    foreach (KeyValuePair<int, double> neighbor in topK)
                    {
                        int uidNeighbor = neighbor.Key;
                        double similarity = neighbor.Value;
                        double iidPosition = itemPositions[uidNeighbor, iid];
                        // TODO: check the standard KNN, we should skip the unseen items somehow!
                        //if (neighborRating != 0)
                        // The weightSum serves as the normalization term
                        // it needs abs() because some metric such as Pearson 
                        // may produce negative weights
                        weightSum += Math.Abs(similarity);
                        weightedSum += iidPosition * similarity;
                    }

                    double position_predicted = weightedSum / weightSum;  // TODO: add some kind of user mean to improve?

                    // TODO: should have a default list of popular items in case of cold users

                    if (topN.Count < topNCount)  // Fill the top N list untill it is full
                    {
                        topN[iid] = position_predicted;
                        if (topN.Count == topNCount)
                        {
                            // Find the item with least position when we have N items in the list
                            min_iid = topN.Aggregate((l, r) => l.Value < r.Value ? l : r).Key;
                            minPosition = topN[min_iid];
                        }
                    }
                    else if (position_predicted > minPosition)
                    {
                        // Replace the least similar neighbor
                        topN.Remove(min_iid);
                        topN[iid] = position_predicted;

                        // Find the item with least position
                        min_iid = topN.Aggregate((l, r) => l.Value < r.Value ? l : r).Key;
                        minPosition = topN[min_iid];
                    }
                }
                // Add the top N items for user uid
                userRecommendations[uid] = topN.Keys.ToList();
            }

            return userRecommendations;
            */
            #endregion
        }
        #endregion
    }
}
