using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace RecSys.Evaluation
{
    /// <summary>
    /// Normalized Discounted Cumulative Gain
    /// See https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain
    /// and here for application in RecSys:
    /// Weimer, M., Karatzoglou, A., Le, Q. V., & Smola, A. (2007). 
    /// Maximum margin matrix factorization for collaborative ranking. NIPS.
    /// </summary>
    public class NCDG
    {
        /// <summary>
        /// Compute the NCDG for pairs of lists, applies to ranking lists.
        /// </summary>
        /// <param name="correctLists">Each Key is a user and Value is the unordered 
        /// set of relevant items of this user.</param>
        /// <param name="predictedLists">Each Key is a user and Value is an ordered  
        /// (descending) list of predicted items for this user.</param>
        /// <param name="topN">Only the topN items in the predictedList will be evaluated.</param>
        /// <returns>The NDCG result in [0,1] the higher the better.</returns>
        /// <remarks>Users with less than topN relevant items will be excluded from calculations.</remarks>
        public static double Evaluate(Dictionary<int, List<int>> correctLists,
            Dictionary<int, List<int>> predictedLists, int topN)
        {
            // The keys in both dictionaries are assumed to be the same
            Debug.Assert(correctLists.Keys.All(predictedLists.Keys.Contains)
                && correctLists.Count == predictedLists.Count);

            double ncdgSum = 0;
            double ncdgCount = 0;

            foreach (int user in correctLists.Keys)
            {
                double ncdg = NDCGAtTopN(correctLists[user], predictedLists[user], topN);

                // If there are fewer relevant items than N then this user 
                // will be skipped from evaluation this is the same as consider 
                // this user has never been included in test set
                // let me know if there is a better treatment
                if (ncdg != -1)
                {
                    ncdgSum += ncdg;
                    ncdgCount++;
                }
            }
            return ncdgSum / ncdgCount;
        }

        // Here we assume all relevant items in the correctList are equally relevant
        #region List wise Evaluate(), all relevance = 1
        /// <summary>
        /// Compute the NDCG between two lists, only the top N items are considered.
        /// The all items in the correctItems are assumed to be equally relevant, i.g. relevance = 1.
        /// See https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain
        /// </summary>
        /// <param name="correctItems">A list contains all relevant items.</param>
        /// <param name="predictedItems">A list contains predicted items, in descending order of item relevance.</param>
        /// <param name="topN">Evaluation will be done on the top N items. If there are less items in the correctItems than
        /// the top N then this case will be skipped.</param>
        /// <returns>The NDCG between two lists. 
        /// The value falls into [0,1] where 1 indicates perfect match.
        /// A -1 value indicates this test is invald as  tehre are less items 
        /// in the correctItems than topN.
        /// </returns>
        private static double NDCGAtTopN(List<int> correctItems, List<int> predictedItems, int topN)
        {
            double relevance = 1.0;
            double dcg = 0;
            double idcg = 0;

            if (correctItems.Count < topN)
            {
                return -1;  // Invalid evaluation
            }

            // The ideal DCG, i.e. all items on the predicted list are on the correct list
            for (int i = 0; i < topN; ++i)
            {
                int rank = i + 1;   // because i starts from 0
                idcg += (Math.Pow(2, relevance) - 1.0) / Math.Log(rank + 1, 2);
            }

            // The DCG of predicted list
            for (int i = 0; i < topN && i < predictedItems.Count; i++)
            {
                int rank = i + 1;   // because i starts from 0
                int indexOfItem = predictedItems[i];

                // If the item is not relevant then relevance = 0. It also implies
                // that dcg will be zero thus we skip the calculation
                if (!correctItems.Contains(indexOfItem)) { continue; }

                dcg += (Math.Pow(2, relevance) - 1.0) / Math.Log(rank + 1, 2);
            }

            return dcg / idcg;
        }
        #endregion
    }
}
