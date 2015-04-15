using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Numerical;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace RecSys.Ordinal
{
    public class PrefRelations
    {

        #region Properties and accessors
        Dictionary<int, SparseMatrix> preferenceRelations;
        private Matrix<double> userSimilarities;

        public Dictionary<int, SparseMatrix> PreferenceRelationsByUser
        {
            get { return preferenceRelations; }
        }

        public PrefRelations(int itemCount)
        {
            preferenceRelations = new Dictionary<int, SparseMatrix>();
            ItemCount = itemCount;
        }

        public SparseMatrix this[int uid]
        {
            get { return preferenceRelations[uid]; }
            set { preferenceRelations[uid] = value; }
        }

        public double this[int uid, int iid_i, int iid_j]
        {
            get { return preferenceRelations[uid][iid_i, iid_j]; }
            set { preferenceRelations[uid][iid_i, iid_j] = value; }
        }

        public List<int> Users
        {
            get { return preferenceRelations.Keys.ToList(); }
        }

        public int UserCount
        {
            get { return preferenceRelations.Count; }
        }

        public int ItemCount
        {
            get;
            set;
        }
        public Matrix<double> UserSimilarities
        {
            get
            {
                //Debug.Assert(!(userSimilarities == null));
                return userSimilarities;
            }
            set
            {
                userSimilarities = value;
            }
        }

        public int GetTotalPrefRelationsCount()
        {
            int count = 0;

            foreach (var pair in PreferenceRelationsByUser)
            {
                SparseMatrix preferenceRelationsOfUser = pair.Value;
                count += preferenceRelationsOfUser.NonZerosCount;
            }
            return count;
        }

        [Obsolete("Used by old implementation.")]
        public Dictionary<int, List<int>> GetSeenItemsByUser()
        {
            Dictionary<int, List<int>> seenItemsByUser = new Dictionary<int, List<int>>();
            Object lockMe = new Object();
            Parallel.ForEach(preferenceRelations, pair =>
            {
                int userIndex = pair.Key;
                SparseMatrix userPreferences = pair.Value;
                List<int> seenItems = new List<int>();
                foreach (Tuple<int, Vector<double>> tuple in userPreferences.EnumerateRowsIndexed())
                {
                    int itemIndex = tuple.Item1;
                    SparseVector itemPreferences = SparseVector.OfVector(tuple.Item2);
                    if (itemPreferences.NonZerosCount > 1) // 1 because the item must has compared with itself
                    {
                        seenItems.Add(itemIndex);
                    }
                }
                lock (lockMe)
                {
                    seenItemsByUser[userIndex] = seenItems;
                }
            });

            return seenItemsByUser;
        }
        #endregion

        #region CreateDiscrete
        public static PrefRelations CreateDiscrete(RatingMatrix R)
        {
            int userCount = R.UserCount;
            int itemCount = R.ItemCount;
            PrefRelations PR = new PrefRelations(itemCount);

            // Create a preference matrix for each user
            Object lockMe = new Object();
            Parallel.ForEach(R.Users, user =>
            {
                int userIndex = user.Item1;
                RatingVector userRatings = new RatingVector(user.Item2);

                Utils.PrintEpoch("Doing user/total", userIndex, userCount);

                // The diagonal refer to the i-i item pair
                SparseMatrix userPreferences = new SparseMatrix(itemCount);

                // The diagonal is left empty!
                //SparseMatrix.OfMatrix(Matrix.Build.SparseDiagonal(itemCount, Config.Preferences.EquallyPreferred));

                // TODO: Use Vector.Map2 to replace the following two foreach loops

                // Here we need to compare each pair of items rated by this user
                foreach (Tuple<int, double> left in userRatings.Ratings)
                {
                    int leftItemIndex = left.Item1;
                    double leftItemRating = left.Item2;

                    foreach (Tuple<int, double> right in userRatings.Ratings)
                    {
                        int rightItemIndex = right.Item1;

                        // TODO: We could compute only the lower triangular, 
                        // and uppwer will be a negative mirror
                        // Let's do it directly at this stage
                        double rightItemRating = right.Item2;

                        Debug.Assert(rightItemRating != 0 && leftItemRating != 0);

                        // Skip the diagonal
                        if (leftItemIndex == rightItemIndex) { continue; }

                        if (leftItemRating > rightItemRating)
                        {
                            userPreferences[leftItemIndex, rightItemIndex] = Config.Preferences.Preferred;
                        }
                        else if (leftItemRating < rightItemRating)
                        {
                            userPreferences[leftItemIndex, rightItemIndex] = Config.Preferences.LessPreferred;
                        }
                        else // i.e. leftItemRating==ratingRight
                        {
                            userPreferences[leftItemIndex, rightItemIndex] = Config.Preferences.EquallyPreferred;
                        }
                    }
                }

                // Because pr's upper triangular should be a mirror of the lower triangular
                Debug.Assert((userPreferences.NonZerosCount).IsEven());
                double debug1 = (Math.Pow(((SparseVector)R.GetRow(userIndex)).NonZerosCount, 2) 
                    - ((SparseVector)R.GetRow(userIndex)).NonZerosCount);
                double debug2 = userPreferences.NonZerosCount;
                Debug.Assert(debug1 == debug2);

                lock (lockMe)
                {
                    // Copy similarity values from lower triangular to upper triangular
                    //pr_uid = DenseMatrix.OfMatrix(pr_uid + pr_uid.Transpose() - DenseMatrix.CreateIdentity(pr_uid.RowCount));
                    PR[userIndex] = userPreferences;
                }
            });



            return PR;
        }
        #endregion

        #region CreateScalar
        // TODO: Scalar preference relations based on Bradley-Terry model
        public static PrefRelations CreateScalar(RatingMatrix R)
        {
            int userCount = R.UserCount;
            int itemCount = R.ItemCount;
            PrefRelations PR = new PrefRelations(itemCount);

            // Create a preference matrix for each user
            Object lockMe = new Object();
            Parallel.ForEach(R.Users, user =>
            {
                int userIndex = user.Item1;
                RatingVector userRatings = new RatingVector(user.Item2);

                Utils.PrintEpoch("Doing user/total", userIndex, userCount);

                // The diagonal refer to the i-i item pair
                SparseMatrix userPreferences = new SparseMatrix(itemCount);

                // The diagonal is left empty!
                //SparseMatrix.OfMatrix(Matrix.Build.SparseDiagonal(itemCount, Config.Preferences.EquallyPreferred));

                // TODO: Use Vector.Map2 to replace the following two foreach loops

                // Here we need to compare each pair of items rated by this user
                foreach (Tuple<int, double> left in userRatings.Ratings)
                {
                    int leftItemIndex = left.Item1;
                    double leftItemRating = left.Item2;

                    foreach (Tuple<int, double> right in userRatings.Ratings)
                    {
                        int rightItemIndex = right.Item1;

                        // TODO: We could compute only the lower triangular, 
                        // and uppwer will be a negative mirror
                        // Let's do it directly at this stage
                        double rightItemRating = right.Item2;

                        Debug.Assert(rightItemRating != 0 && leftItemRating != 0);

                        // Skip the diagonal
                        if (leftItemIndex == rightItemIndex) { continue; }

                        userPreferences[leftItemIndex, rightItemIndex] = 0.1 * (leftItemRating - rightItemRating + 5);//(double)leftItemRating / (leftItemRating + rightItemRating);
                    }
                }

                // Because pr's upper triangular should be a mirror of the lower triangular
                Debug.Assert((userPreferences.NonZerosCount).IsEven());
                double debug1 = (Math.Pow(((SparseVector)R.GetRow(userIndex)).NonZerosCount, 2)
                    - ((SparseVector)R.GetRow(userIndex)).NonZerosCount);
                double debug2 = userPreferences.NonZerosCount;
                Debug.Assert(debug1 == debug2);

                lock (lockMe)
                {
                    // Copy similarity values from lower triangular to upper triangular
                    //pr_uid = DenseMatrix.OfMatrix(pr_uid + pr_uid.Transpose() - DenseMatrix.CreateIdentity(pr_uid.RowCount));
                    PR[userIndex] = userPreferences;
                }
            });



            return PR;
        }
        #endregion

        #region PreferencesToPositions
        /// <summary>
        /// Convert one user's preference relations into position matrix
        /// See: Brun, A., Hamad, A., Buffet, O., & Boyer, A. (2010). 
        /// Towards preference relations in recommender systems. Workshop in ECML-PKDD.
        /// </summary>
        /// <param name="userPreferences"></param>
        /// <returns></returns>
        public Vector<double> PreferencesToPositions(SparseMatrix userPreferences)
        {
            // Count for each preference type
            // Actually the original paper count the strict preferred and less preferred by exact match
            // but here we just compare with the EquallyPreferred, because the preferences can be 
            // scalar with Bradley-Terry model. However, it won't affect the result when it is discrete preference relations.
            SparseVector preferredCountByItem = SparseVector.OfEnumerable(userPreferences.FoldByRow((count, pref) =>
                    count + (pref == Config.Preferences.Preferred ? 1 : 0), 0.0));

            SparseVector lessPreferredCountByItem = SparseVector.OfEnumerable(userPreferences.FoldByRow((count, pref) =>
                    count + (pref == Config.Preferences.LessPreferred ? 1 : 0), 0.0));

            // Note that we assume the diagonal are left empty
            // otherwise the equally count needs to be offset by 1 for each, i.e. item itself does not count
            Debug.Assert(userPreferences.Trace() == 0);
            SparseVector equallyPreferredCountByItem = SparseVector.OfEnumerable(userPreferences.FoldByRow((count, pref) =>
                    count + (pref == Config.Preferences.EquallyPreferred  ? 1 : 0), 0.0));

            // Note that if the position is value zero then it won't appear in  positionByItem
            // because the use of SparseVector.OfVector() will ignore all zero values
            Vector<double> positionByItem =
     (preferredCountByItem - lessPreferredCountByItem)
     .PointwiseDivide(lessPreferredCountByItem + preferredCountByItem + equallyPreferredCountByItem) + Config.Preferences.PositionShift;

            Vector<double> indicatorOfSeenItems = userPreferences.RowSums();    // If zero, then this item has never been seen
            for(int i = 0; i < indicatorOfSeenItems.Count; i++)
            {
                if(indicatorOfSeenItems[i] == 0)
                {
                    positionByItem[i] = SparseVector.Zero;
                }
            }

            //+Config.Preferences.PositionShift;

            // TODO: May improve later. Some items have position 0 and we dont want to mix
            // up the position 0 and the 0 in sparsematrix.
            // So we use a constant to hold the space for position value 0
            // it should be reverted back when use
            //Vector<double> indicatorVector = userPreferences.RowSums();
            //for (int i = 0; i < indicatorVector.Count; i++)
            //{
                //if (indicatorVector[i] != 0 && positionByItem[i] != 0)
                //{
                //    Debug.Assert(true, "By using the PositionShift constant, we should not be in here.");
                 //   positionByItem[i] = Config.ZeroInSparseMatrix;
               // }
            //}

            return positionByItem;
        }
        #endregion

        #region GetPositionMatrix

        // The entry position_ij is the position value of item j for user i
        public SparseMatrix GetPositionMatrix()
        {
            SparseMatrix positionMatrix = new SparseMatrix(UserCount, ItemCount);
            Dictionary<int, Vector<double>> positionsByUser = new Dictionary<int, Vector<double>>(UserCount);

            // For each user
            Object lockMe = new Object();
            Parallel.ForEach(preferenceRelations, pair =>
            {
                int indexOfUser = pair.Key;
                SparseMatrix preferencesOfUser = pair.Value;

                // Convernt preferences into positions
                Vector<double> positionsOfUser = PreferencesToPositions(preferencesOfUser);
                lock (lockMe)
                {
                    positionsByUser[indexOfUser] = positionsOfUser;
                }
            });

            // Order the position vectors by user index
            var vectorsOfPositionSortedByUser = from pair in positionsByUser
                                                orderby pair.Key ascending
                                                select pair.Value;

            positionMatrix = SparseMatrix.OfRowVectors(vectorsOfPositionSortedByUser);
            return positionMatrix;
        }
        #endregion

        #region GetRecommendations
        /// <summary>
        /// Get the topN items for each user
        /// </summary>
        /// <param name="topN"></param>
        /// <returns>Dictionary with user index as the key and top N items
        /// as the values for this user.</returns>
        public Dictionary<int, List<int>> GetTopNItemsByUser(int topN)
        {
            int userCount = this.UserCount;
            int itemCount = this.ItemCount;
            Dictionary<int, List<int>> topNItemsByUser = new Dictionary<int, List<int>>(userCount);
            Dictionary<int, List<int>> topNItemsByUserNew = new Dictionary<int, List<int>>(userCount);

            // Compute the topN list for each user
            Object lockMe = new Object();
            Parallel.ForEach(preferenceRelations, pair =>
            {
                // topN stores the top N item IDs and positions
                Dictionary<int, double> topNItems = new Dictionary<int, double>(topN);
                int userIndex = pair.Key;
                Utils.PrintEpoch("Get topN items for user/total", userIndex, userCount);
                SparseMatrix userPreferences = pair.Value;

                #region The old correct but slow version
                /*

                // Compute the position of each item
                double minPosition = double.MinValue;
                int min_iid = int.MinValue;
                double maxPosition = 0.0;
                foreach (Tuple<int, Vector<double>> row in userPreferences.EnumerateRowsIndexed())
                {
                    int iid = row.Item1;
                    SparseVector preferences = SparseVector.OfVector(row.Item2);
                    // TODO: check it, I expect the best item to have the highest position
                    int preferredCount = 0;
                    int lessPreferredCount = 0;
                    foreach (double preference in preferences.Enumerate(Zeros.AllowSkip))
                    {
                        if (preference == Config.Preferences.Preferred) ++preferredCount;
                        else if (preference == Config.Preferences.LessPreferred) ++lessPreferredCount;
                    }
                    double iid_position = (double)(lessPreferredCount-preferredCount)/preferences.NonZerosCount;
                    if (iid_position > maxPosition) maxPosition = iid_position;
                    // All items are added into the topN list if it is not full
                    // otherwise the least positioned item will be replaced
                    if (topNItems.Count < topN)
                    {
                        topNItems[iid] = iid_position;
                        if (topNItems.Count == topN)
                        {
                            min_iid = topNItems.Aggregate((l, r) => l.Value < r.Value ? l : r).Key;
                            minPosition = topNItems[min_iid];
                        }
                    }
                    else if (iid_position > minPosition)
                    {
                        // Replace the least positioned item
                        topNItems.Remove(min_iid);
                        // Add the current item
                        topNItems[iid] = iid_position;

                        // Find the item with least position
                        min_iid = topNItems.Aggregate((l, r) => l.Value < r.Value ? l : r).Key;
                        minPosition = topNItems[min_iid];
                    }

                    if (iid == 21 && userIndex == 0)
                    {
                        Console.WriteLine("Position of 21 = " + iid_position);
                    }
                    if (iid == 141 && userIndex == 0)
                    {
                        int a = preferenceRelations[userIndex].Row(141).Count(x=>x==Config.Preferences.Preferred?true:false);
                        int b =preferenceRelations[userIndex].Row(141).Count(x => x == Config.Preferences.LessPreferred? true : false);
                        int c = a - b;
                        Console.WriteLine("Position of 141 = " + iid_position);
                    }
                }

                // Get the keys (iid) in the topN list sorted by positions
                List<int> sortedItems = new List<int>(from entry in topNItems orderby entry.Value ascending select entry.Key);
                lock (lockMe)
                {
                    topNItemsByUser[userIndex] = sortedItems;
                }
                */
                #endregion

                // This gives me an array of how many preferred for each item
                Vector<double> preferredCountByItem = DenseVector.OfArray(
                    userPreferences.FoldByRow((count, rating) =>
                        count + (rating == Config.Preferences.Preferred ? 1 : 0), 0.0));
                Vector<double> lessPreferredCountByItem = DenseVector.OfArray(
                userPreferences.FoldByRow((count, rating) =>
                count + (rating == Config.Preferences.LessPreferred ? 1 : 0), 0.0));
                Vector<double> equallyPreferredCountByItem = DenseVector.OfArray(
userPreferences.FoldByRow((count, rating) =>
count + (rating == Config.Preferences.EquallyPreferred ? 1 : 0), 0.0));

                List<double> positionByItem = (preferredCountByItem - lessPreferredCountByItem).PointwiseDivide(
                    lessPreferredCountByItem + preferredCountByItem + equallyPreferredCountByItem
                    ).ToList();
                List<int> itemIndexSortedByPosition = Enumerable.Range(0, positionByItem.Count).ToList();
                Sorting.Sort(positionByItem, itemIndexSortedByPosition);
                positionByItem.Reverse();   // This is now the sorted position values
                itemIndexSortedByPosition.Reverse();    // This is now the the item index sorted by position

                // LINQ version
                //var sorted = positionByItem
                //    .Select((x, i) => new KeyValuePair<double, int>(x, i))
                //.OrderBy(x => x.Key)
                //.ToList();
                //List<double> B = sorted.Select(x => x.Key).ToList();
                //List<int> idx = sorted.Select(x => x.Value).ToList();

                lock (lockMe)
                {
                    topNItemsByUserNew[userIndex] = itemIndexSortedByPosition.GetRange(0, topN);
                    // The corresponding top N position values are in positionByItem.GetRange(0, topN);
                }
            });

            return topNItemsByUser;
        }
        #endregion


        /// <summary>
        /// Map the original values into several equal size bins,
        /// a new value is assigned depending on the bin
        /// </summary>
        /// <param name="range"></param>
        /// <param name="binCount"></param>
        /// <param name="quantilizer"></param>
        public void Quantilize(double range, int binCount, List<double> quantilizer)
        {
            Dictionary<int, SparseMatrix> preferenceRelationsQuantilized
                = new Dictionary<int, SparseMatrix>(preferenceRelations.Count);

            double binSize = range / binCount;
            Utils.WriteMatrix(preferenceRelations[0], "beforeQuantilize.csv");
            foreach(var pair in preferenceRelations)
            {
                int indexOfUser = pair.Key;
                SparseMatrix preferencesOfUser = pair.Value;
                SparseMatrix preferencesOfUserQuantilized = new SparseMatrix(preferencesOfUser.RowCount);
                foreach (var element in preferencesOfUser.EnumerateIndexed(Zeros.AllowSkip))
                {
                    int indexOfItem_i = element.Item1;
                    int indexOfItem_j = element.Item2;
                    double preference = element.Item3;
                    for (int indexOfBin = 0; indexOfBin < binCount; indexOfBin++)
                    {
                        if (preference < (indexOfBin + 1) * binSize)
                        {
                            preferencesOfUserQuantilized[indexOfItem_i, indexOfItem_j] = quantilizer[indexOfBin];
                            break;
                        }
                    }

                    preferenceRelationsQuantilized[indexOfUser] = preferencesOfUserQuantilized;
                }
            }

            preferenceRelations = preferenceRelationsQuantilized;
            Utils.WriteMatrix(preferenceRelations[0], "afterQuantilize.csv");
        }
    }
}
