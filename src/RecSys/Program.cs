using RecSys.Numerical;
using RecSys.Ordinal;
using System;
using System.Collections.Generic;
using System.Linq;
using RecSys.Core;
using RecSys.Evaluation;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra.Mkl;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;

namespace RecSys
{
    class Program
    {
        static void Main(string[] args)
        {
            // Enable multi-threading for Math.Net
            Control.UseMultiThreading();

            /************************************************************
             *   1. Load data sets
             *   2. Compute/load similarities
             *   3. R_train     => Rating Matrix train set
             *   4. R_test      => Rating Matrix test set
             *   5. R_unknown   => Rating Matrix with ones indicating unknown entries in the R_test
             *   6. PR_train    => Preference relations constructed from R_train
             *   7. userSimilaritiesOfRating    => The user-user similarities from R_train
             *   8. userSimilaritiesOfPref      => The user-user similarities from PR_train
             *   9. relevantItemsByUser         => The relevant items of each user based on R_test, 
             *          is used as ground truth in all ranking evalution
            ************************************************************/
            #region Prepare rating data
            Utils.StartTimer();
            RatingMatrix R_train;
            RatingMatrix R_test;

            // Load saved or fresh data
            if(Config.LoadSavedData)
            {
                Utils.PrintHeading("Load train/test sets from saved files");
                R_train = new RatingMatrix(Utils.ReadSparseMatrix(Config.Ratings.TrainSetFile));
                R_test = new RatingMatrix(Utils.ReadSparseMatrix(Config.Ratings.TestSetFile));
            }
            else
            {
                Utils.PrintHeading("Create train/test sets from fresh data");
                Utils.LoadMovieLensSplitByCount(Config.Ratings.DataSetFile, out R_train, out R_test);
                Utils.WriteMatrix(R_train.Matrix, Config.Ratings.TrainSetFile);
                Utils.WriteMatrix(R_test.Matrix, Config.Ratings.TestSetFile);
            }
            
            RatingMatrix R_unknown = R_test.IndexesOfNonZeroElements();
            Console.WriteLine(R_train.DatasetBrief("Train set"));
            Console.WriteLine(R_test.DatasetBrief("Test set"));

            Utils.PrintValue("Relevant item threshold", Config.Ratings.RelevanceThreshold.ToString("0.0"));
            Dictionary<int, List<int>> relevantItemsByUser = ItemRecommendationCore
                .GetRelevantItemsByUser(R_test, Config.Ratings.RelevanceThreshold);
            Utils.PrintValue("Mean # of relevant items per user",
                relevantItemsByUser.Average(k => k.Value.Count).ToString("0"));
            Utils.StopTimer();
            Utils.Pause();
            #endregion

            #region Prepare preference relation data
            Utils.StartTimer();
            Utils.PrintHeading("Prepare preferecen relation data");
            PrefRelations PR_train = PrefRelations.CreateDiscrete(R_train);
            Utils.StopTimer();
            #endregion

            #region Compute or load similarities
            DenseMatrix userSimilaritiesOfRating;
            DenseMatrix userSimilaritiesOfPref;
            if (Config.LoadSavedData)
            {
                Utils.StartTimer();
                Utils.PrintHeading("Load user-user similarities from R_train");
                userSimilaritiesOfRating = Utils.ReadDenseMatrix(Config.Ratings.UserSimilaritiesOfRatingFile);
                Utils.PrintValue("Sum of similarities", userSimilaritiesOfRating.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", userSimilaritiesOfRating.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Load user-user similarities from PR_train");
                userSimilaritiesOfPref = Utils.ReadDenseMatrix(Config.Ratings.UserSimilaritiesOfPrefFile);
                Utils.PrintValue("Sum of similarities", userSimilaritiesOfPref.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", userSimilaritiesOfPref.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();
            }
            else
            {
                Utils.StartTimer();
                Utils.PrintHeading("Compute user-user similarities from R_train");
                userSimilaritiesOfRating = Metric.GetPearsonOfRows(R_train);
                Utils.WriteMatrix(userSimilaritiesOfRating, Config.Ratings.UserSimilaritiesOfRatingFile);
                Utils.PrintValue("Sum of similarities", userSimilaritiesOfRating.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", userSimilaritiesOfRating.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Compute user-user similarities from PR_train");
                userSimilaritiesOfPref = Metric.GetCosineOfPrefRelations(PR_train);
                Utils.WriteMatrix(userSimilaritiesOfPref, Config.Ratings.UserSimilaritiesOfPrefFile);
                Utils.PrintValue("Sum of similarities", userSimilaritiesOfPref.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", userSimilaritiesOfPref.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();
            }
            R_train.UserSimilarities = userSimilaritiesOfRating;
            PR_train.UserSimilarities = userSimilaritiesOfPref;

            #endregion


            /************************************************************
             *   Global Mean
            ************************************************************/
            #region Run Global Mean
            if (Config.RunGlobalMean)
            {
                // Prediction
                Utils.PrintHeading("Global Mean");
                Utils.StartTimer();
                double globalMean = R_train.GetGlobalMean();
                RatingMatrix R_predicted = R_unknown.Multiply(globalMean);
                Utils.StopTimer();

                // Evaluation
                Utils.PrintValue("RMSE", RMSE.Evaluate(R_test, R_predicted).ToString("0.0000"));
                Utils.PrintValue("MAE", MAE.Evaluate(R_test, R_predicted).ToString("0.0000"));

                Utils.Pause();
            }
            #endregion

            /************************************************************
             *   Preferecen relations based Non-negative Matrix Factorization
            ************************************************************/
            #region Run preferecen relations based PrefNMF
            if (Config.RunPrefNMF)
            {
                // Prediction
                Utils.PrintHeading("Preferecen relations based PrefNMF");
                Utils.StartTimer();
                RatingMatrix R_predicted = PrefNMF.PredictRatings(PR_train, R_unknown, Config.PrefNMF.MaxEpoch,
                    Config.PrefNMF.LearnRate, Config.PrefNMF.Regularization, Config.PrefNMF.K);
                Utils.StopTimer();

                // Evaluation
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, Config.TopN);
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("Precision@" + n, Precision.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }

                Utils.Pause();
            }
            #endregion


            /************************************************************
             *   Rating based UserKNN
            ************************************************************/
            #region Run rating based UserKNN
            if (Config.RunRatingUserKNN)
            {
                // Prediction
                Utils.PrintHeading("Rating based User KNN");
                Utils.StartTimer();
                RatingMatrix R_predicted = UserKNN.PredictRatings(R_train, R_unknown, Config.KNN.K);
                Utils.StopTimer();

                // Evaluation
                Utils.PrintValue("RMSE", RMSE.Evaluate(R_test, R_predicted).ToString("0.0000"));
                Utils.PrintValue("MAE", MAE.Evaluate(R_test, R_predicted).ToString("0.0000"));
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, Config.TopN);
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("Precision@" + n, Precision.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }

                Utils.Pause();
            }
            #endregion

            /************************************************************
             *   Rating based Non-negative Matrix Factorization
            ************************************************************/
            #region Run rating based NMF
            if (Config.RunNMF)
            {
                // Prediction
                Utils.PrintHeading("Rating based NMF");
                Utils.StartTimer();
                RatingMatrix R_predicted = NMF.PredictRatings(R_train, R_unknown, Config.NMF.MaxEpoch,
                    Config.NMF.LearnRate, Config.NMF.Regularization, Config.NMF.K);
                Utils.StopTimer();

                // Evaluation
                Utils.PrintValue("RMSE", RMSE.Evaluate(R_test, R_predicted).ToString("0.0000"));
                Utils.PrintValue("MAE", MAE.Evaluate(R_test, R_predicted).ToString("0.0000"));
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, Config.TopN);
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("Precision@" + n, Precision.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }

                Utils.Pause();
            }
            #endregion

            /************************************************************
             *   Preference relation based UserKNN
            ************************************************************/
            #region Run preference relation based UserKNN
            if (Config.RunPreferenceUserKNN)
            {
                // Prediction
                Utils.PrintHeading("Preference relation based UserKNN");
                Utils.StartTimer();
                RatingMatrix PR_predicted = PrefUserKNN.PredictRatings(PR_train, R_unknown, Config.KNN.K);
                Utils.StopTimer();

                // Evaluation
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(PR_predicted, Config.TopN);
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("Precision@" + n, Precision.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }

                Utils.Pause();
            }
            #endregion

        }
    }
}
