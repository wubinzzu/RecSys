using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using RecSys.Numerical;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RecSys.Core
{
    /// <summary>
    /// This class implements core functions for item recommendations.
    /// </summary>
    public class ItemRecommendationCore
    {
        #region GetTopNItemsByUser

        /// <summary>
        /// Get the top N items of each user.
        /// The selection is based on the values stored in the matrix,
        /// where an item with higher value is considered as better.
        /// </summary>
        /// <param name="R">The user-by-item matrix, 
        /// each value indicates the quality of the item to a user.</param>
        /// <param name="topN">The number of items to be recommended for each user.</param>
        /// <returns>Each Key is a user and Value is the top N recommended items for that user.</returns>
        public static Dictionary<int, List<int>> GetTopNItemsByUser(RatingMatrix R, int topN)
        {
            int userCount = R.UserCount;
            int itemCount = R.ItemCount;
            Dictionary<int, List<int>> topNItemsByUser = new Dictionary<int, List<int>>(userCount);

            // Select top N items for each user
            foreach (Tuple<int, Vector<double>> user in R.Users)
            {
                int indexOfUser = user.Item1;

                // To be sorted soon
                List<double> ratingsOfItemsSortedByRating = user.Item2.ToList();

                // TODO: this is important, because some models like PrefNMF may produce
                // negative scores, without setting the 0 to negative infinity these
                // items will be ranked before the test items and they are always not relevant items!
                ratingsOfItemsSortedByRating.ForEach(x => x = x == 0 ? double.NegativeInfinity : x);
                List<int> indexesOfItemsSortedByRating = Enumerable.Range(0, ratingsOfItemsSortedByRating.Count).ToList();

                // Sort by rating
                Sorting.Sort<double, int>(ratingsOfItemsSortedByRating, indexesOfItemsSortedByRating);

                // Make it descending order by rating
                ratingsOfItemsSortedByRating.Reverse();
                indexesOfItemsSortedByRating.Reverse();

                topNItemsByUser[indexOfUser] = indexesOfItemsSortedByRating.GetRange(0, topN);

                // In case the ratings of the top N items need to be stored
                // in the future, implement the following:
                //for (int i = 0; i < topN; ++i)
                //{
                // ratingsOfItemsSortedByRating[i] is the rating of the ith item in topN list
                // indexesOfItemsSortedByRating[i] is the index (in the R) of the ith item in topN list
                //}
            }

            return topNItemsByUser;
        }
        #endregion

        #region GetRelevantItemsByUser
        // Get the relevant items of each user, i.e. rated no lower than the criteria
        public static Dictionary<int, List<int>> GetRelevantItemsByUser(RatingMatrix R, double criteria)
        {
            int userCount = R.UserCount;
            int itemCount = R.ItemCount;
            Dictionary<int, List<int>> relevantItemsByUser = new Dictionary<int, List<int>>(userCount);

            // Select relevant items for each user
            foreach (Tuple<int, Vector<double>> user in R.Users)
            {
                int userIndex = user.Item1;
                RatingVector userRatings = new RatingVector(user.Item2);
                List<int> relevantItems = new List<int>();

                foreach (Tuple<int, double> element in userRatings.Ratings)
                {
                    int itemIndex = element.Item1;
                    double rating = element.Item2;
                    if (rating >= criteria)
                    {
                        // This is a relevant item
                        relevantItems.Add(itemIndex);
                    }
                }
                relevantItemsByUser[userIndex] = relevantItems;
            }

            return relevantItemsByUser;
        }
        #endregion
    }
}
