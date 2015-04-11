using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using RecSys;
using RecSys.Core;
using RecSys.Evaluation;
using RecSys.Numerical;
using RecSys.Ordinal;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace RecSysUnitTest
{
    [TestClass]
    public class RecSysTests
    {
        #region Data for testing
        private static RatingMatrix GetSampleRatingMatrix()
        {
            /*
             * 5  3  0  1
             * 4  0  0  1
             * 1  1  0  5
             * 1  0  0  4
             * 0  1  5  4
             */
            RatingMatrix R = new RatingMatrix(new SparseMatrix(5, 4));
            R[0, 0] = 5;
            R[0, 1] = 3;
            R[0, 3] = 1;
            R[1, 0] = 4;
            R[1, 3] = 1;
            R[2, 0] = 1;
            R[2, 1] = 1;
            R[2, 3] = 5;
            R[3, 0] = 1;
            R[3, 3] = 4;
            R[4, 1] = 1;
            R[4, 2] = 5;
            R[4, 3] = 4;

            return R;
        }
        #endregion

        #region Tests on Evaluation package
        [TestClass]
        public class EvaluationTest
        {
            #region RMSETest()
            [TestMethod]
            public void RMSETest()
            {
                /*
                 * 5  3  0  1
                 * 4  0  0  1
                 * 1  1  0  5
                 * 1  0  0  4
                 * 0  1  5  4
                 */
                RatingMatrix R = GetSampleRatingMatrix();

                /*
                 * 3  3  0  1
                 * 4  0  0  5
                 * 1  2  0  5
                 * 1  0  0  4
                 * 0  1  5  4
                 */
                RatingMatrix R_predicted = GetSampleRatingMatrix();
                R_predicted[0, 0] = 3;  // was 5
                R_predicted[2, 1] = 2;  // was 1
                R_predicted[1, 3] = 5;  // was 1

                double debug1 = RMSE.Evaluate(R, R_predicted);
                double debug2 = Math.Sqrt(21.0 / 13.0);
                Debug.Assert(debug1 == debug2);
            }
            #endregion
        }
        #endregion

        #region Tests on PreferenceRelations class
        [TestClass]
        public class PreferenceRelationnsTest
        {
            #region Create discrete preference relations using rating matrix
            [TestMethod]
            public void CreateDiscrete()
            {
                /*
                 * 5  3  0  1
                 * 4  0  0  1
                 * 1  1  0  5
                 * 1  0  0  4
                 * 0  1  5  4
                 */
                RatingMatrix R = GetSampleRatingMatrix();

                // act
                PreferenceRelations PR = PreferenceRelations.CreateDiscrete(R);

                // assert
                foreach (KeyValuePair<int, SparseMatrix> user in PR.GetAllPreferenceRelations)
                {
                    int indexOfUser = user.Key;
                    SparseMatrix preferencesOfUser = user.Value;

                    // Note that the diagonal (item compares to itsself) is elft empty
                    Debug.Assert(preferencesOfUser.Trace() == 0);

                    // Check if the correct number of preference relations have been created
                    Debug.Assert((Math.Pow(R.GetRow(indexOfUser).NonZerosCount, 2)
                        - R.GetRow(indexOfUser).NonZerosCount)
                        == preferencesOfUser.NonZerosCount);
                }

                // Check if the first user's preferences are correct
                Debug.WriteLine("PR[0][0, 0]=" + PR[0][0, 0]);
                Debug.Assert(PR[0][0, 0] == SparseMatrix.Zero);
                Debug.Assert(PR[0][0, 1] == Config.Preferences.Preferred);
                Debug.Assert(PR[0][1, 0] == Config.Preferences.LessPreferred);
                Debug.Assert(PR[0][0, 2] == SparseMatrix.Zero);
                Debug.Assert(PR[0][2, 0] == SparseMatrix.Zero);
                Debug.Assert(PR[0][1, 2] == SparseMatrix.Zero);
                Debug.Assert(PR[0][2, 1] == SparseMatrix.Zero);
                Debug.Assert(PR[0][1, 3] == Config.Preferences.Preferred);
                Debug.Assert(PR[0][3, 1] == Config.Preferences.LessPreferred);

                // Check if the last user's preferences are correct
                Debug.Assert(PR[4][1, 1] == SparseMatrix.Zero);
                Debug.Assert(PR[4][0, 1] == SparseMatrix.Zero);
                Debug.Assert(PR[4][1, 0] == SparseMatrix.Zero);
                Debug.Assert(PR[4][0, 2] == SparseMatrix.Zero);
                Debug.Assert(PR[4][2, 0] == SparseMatrix.Zero);
                Debug.Assert(PR[4][1, 2] == Config.Preferences.LessPreferred);
                Debug.Assert(PR[4][2, 1] == Config.Preferences.Preferred);
                Debug.Assert(PR[4][1, 3] == Config.Preferences.LessPreferred);
                Debug.Assert(PR[4][3, 1] == Config.Preferences.Preferred);
            }
            #endregion
        }
        #endregion

        #region Tests on RatingMatrix class
        [TestClass]
        public class RatingMatrixTest
        {
            [TestMethod]
            public void GetTopNItemsByUser()
            {
                /*
                 * 5  3  0  1
                 * 4  0  0  1
                 * 1  1  0  5
                 * 1  0  0  4
                 * 0  1  5  4
                 */
                RatingMatrix R = GetSampleRatingMatrix();

                // act
                Dictionary<int, List<int>> topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R, 2);

                // assert
                Debug.Assert(topNItemsByUser[0].Count == 2);
                Debug.Assert(topNItemsByUser[0][0] == 0);
                Debug.Assert(topNItemsByUser[0][1] == 1);
                Debug.Assert(topNItemsByUser[1][0] == 0);
                Debug.Assert(topNItemsByUser[1][1] == 3);
                Debug.Assert(topNItemsByUser[4][0] == 2);
                Debug.Assert(topNItemsByUser[4][1] == 3);
            }
        }
        #endregion

        #region Tests on Utilities class
        [TestClass]
        public class UtilitiesTest
        {
            [TestMethod]
            public void LoadMovieLens()
            {
                // act
                RatingMatrix R_train;
                RatingMatrix R_test;
                Utils.LoadMovieLensSplitByCount("ua.test", out R_train, out R_test, 9, 7);
                // 1	265	4	878542441
                // assert
                Debug.Assert(R_train.GetRow(0).NonZerosCount == 7);
                Debug.Assert(R_train.GetRow(1).NonZerosCount == 7);
                Debug.Assert(R_train.GetRow(2).NonZerosCount == 7);

                Debug.Assert(R_test.GetRow(0).NonZerosCount == 2);
                Debug.Assert(R_test.GetRow(1).NonZerosCount == 3);
                Debug.Assert(R_test.GetRow(2).NonZerosCount == 3);

                Debug.Assert(R_train.GetRow(0)[6] == 5);
                Debug.Assert(R_test.GetRow(0)[6] == SparseMatrix.Zero);
                Debug.Assert(R_test.GetRow(0)[7] == 3);
                Debug.Assert(R_train.GetRow(0)[7] == SparseMatrix.Zero);
                Debug.Assert(R_test.GetRow(0)[8] == 5);
            }
        }
        #endregion

        #region Tests on PrefUserKNN
        [TestClass]
        public class PrefUserKNNTest
        {
            #region Convert one user's preference relations into positions
            [TestMethod]
            public void PreferencesToPositions()
            {
                /*
                 * 5  3  0  1
                 * 4  0  0  1
                 * 1  1  0  5
                 * 1  0  0  4
                 * 0  1  5  4
                 */
                RatingMatrix R = GetSampleRatingMatrix();
                PreferenceRelations PR = PreferenceRelations.CreateDiscrete(R);

                // act
                // Convert first, Third, and last users' preferences to positions
                SparseVector positionsOfUserFirst = PR.PreferencesToPositions(PR[0]);
                SparseVector positionsOfUserThird = PR.PreferencesToPositions(PR[2]);
                SparseVector positionsOfUserLast = PR.PreferencesToPositions(PR[4]);

                // assert
                // Check first user
                Debug.Assert(positionsOfUserFirst[0] == 1);
                Debug.Assert(positionsOfUserFirst[1] == Config.ZeroInSparseMatrix); // It is actually a value 0
                Debug.Assert(positionsOfUserFirst[2] == SparseMatrix.Zero);
                Debug.Assert(positionsOfUserFirst[3] == -1);

                // Check third user
                Debug.Assert(positionsOfUserThird[0] == -0.5);
                Debug.Assert(positionsOfUserThird[1] == -0.5);
                Debug.Assert(positionsOfUserThird[2] == SparseMatrix.Zero);
                Debug.Assert(positionsOfUserThird[3] == 1);

                // Check second last user
                Debug.Assert(positionsOfUserLast[0] == SparseMatrix.Zero);
                Debug.Assert(positionsOfUserLast[1] == -1);
                Debug.Assert(positionsOfUserLast[2] == 1);
                Debug.Assert(positionsOfUserLast[3] == Config.ZeroInSparseMatrix); // It is actually a value 0

                // The number of positions should match the number of ratings by each user
                Debug.Assert(positionsOfUserFirst.NonZerosCount
                    == SparseVector.OfVector(R.Matrix.Row(0)).NonZerosCount, String.Format("{0}=={1}",
                    positionsOfUserFirst.NonZerosCount, SparseVector.OfVector(R.Matrix.Row(0)).NonZerosCount));

                Debug.Assert(positionsOfUserThird.NonZerosCount
                    == SparseVector.OfVector(R.Matrix.Row(2)).NonZerosCount, String.Format("{0}=={1}",
                    positionsOfUserThird.NonZerosCount, SparseVector.OfVector(R.Matrix.Row(2)).NonZerosCount));

                Debug.Assert(positionsOfUserLast.NonZerosCount
                    == SparseVector.OfVector(R.Matrix.Row(4)).NonZerosCount);
            }
            #endregion

            #region Convert all users' preference relations into a single position matrix
            [TestMethod]
            public void GetPositionMatrix()
            {
                /*
                 * 5  3  0  1
                 * 4  0  0  1
                 * 1  1  0  5
                 * 1  0  0  4
                 * 0  1  5  4
                 */
                RatingMatrix R = GetSampleRatingMatrix();
                PreferenceRelations PR = PreferenceRelations.CreateDiscrete(R);

                // act
                SparseMatrix positionMatrix = PR.GetPositionMatrix();

                // assert
                // How many ratings we have then how many positions we have
                Debug.Assert(positionMatrix.NonZerosCount == R.Matrix.NonZerosCount);

                // Check if each rating has a corresponding position
                // we have check the count so don't need to check the oppsite
                foreach (Tuple<int, int, double> element in R.Matrix.EnumerateIndexed(Zeros.AllowSkip))
                {
                    int indexOfUser = element.Item1;
                    int indexOfItem = element.Item2;
                    double rating = element.Item3;

                    Debug.Assert(positionMatrix[indexOfUser, indexOfItem] != SparseMatrix.Zero);
                }
            }
            #endregion

            #region Generate top N recommendation for each user
            [TestMethod]
            public void RecommendTopN()
            {
                /*
                 * 5  3  0  1
                 * 4  0  0  1
                 * 1  1  0  5
                 * 1  0  0  4
                 * 0  1  5  4
                 */
                RatingMatrix R = GetSampleRatingMatrix();
                PreferenceRelations PR = PreferenceRelations.CreateDiscrete(R);
                List<int> targetUsers = new List<int> { 0, 1, 2, 3, 4 };

                /*
                       1  1E-14  0     -1
                       1      0  0     -1
                    -0.5   -0.5  0      1
                      -1      0  0      1
                       0     -1  1  1E-14
                 */
                SparseMatrix positionMatrix = PR.GetPositionMatrix();


                /*
                             1   0.774291  -0.186441  -0.178683  -0.978839
                      0.774291          1  0.0198536   0.162791  -0.628768
                     -0.186441  0.0198536          1   0.972828   0.221028
                     -0.178683   0.162791   0.972828          1   0.258904
                     -0.978839  -0.628768   0.221028   0.258904          1
                 */
                DenseMatrix userSimilarities = R.UserPearson();

                // act
                PR.UserSimilarities = userSimilarities;
                Dictionary<int, List<int>> topNItemsByUser = PrefUserKNN.RecommendTopN(PR, 3, targetUsers, 4);

                // assert
                // Check the topN list of the first user
                // the top 3 neighbors of user index 0 are user indexes 1->3->2
                // so the predicted position of user index 0 on item 0 is = 0.774291 * positionMatrix[0,0] -0.178683 * positionMatrix[0,0]
                //double correctPositionOfItem0User0 = ((0.774291 * 1)+ (-0.178683 * -1) + (-0.186441 * -0.5)) / (0.774291 - 0.186441 - 0.178683);
                //double correctPositionOfItem1User0 = ((-0.186441 * -0.5)) / (- 0.178683);
                //double correctPositionOfItem2User0 = 0;
                //double correctPositionOfItem3User0 = ((0.774291 * -1)+ (-0.178683 * 1) + (-0.186441 * 1)) / (0.774291 - 0.186441 - 0.178683);
                Debug.Assert(topNItemsByUser[0][0] == 0);
                Debug.Assert(topNItemsByUser[0][1] == 2);
                Debug.Assert(topNItemsByUser[0][2] == 1);
                Debug.Assert(topNItemsByUser[0][3] == 3);
            }
            #endregion
        }
        #endregion

    }
}
