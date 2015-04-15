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
using System.IO;

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
            PrefRelations PR_test = PrefRelations.CreateDiscrete(R_test);
            //PrefRelations PR_train = PrefRelations.CreateScalar(R_train);
            Utils.StopTimer();
            #endregion

            #region Compute or load similarities
            Matrix<double> userSimilaritiesOfRating;
            Matrix<double> userSimilaritiesOfPref;
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
             *   Rating based Non-negative Matrix Factorization
            ************************************************************/
            #region Run rating based NMF
            Utils.PrintHeading("Rating based NMF");
            if (Utils.Ask())
            {
                // Prediction
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

                //Utils.Pause();
            }
            #endregion


            /************************************************************
             *   Ordinal Matrix Factorization with NMF as scorer
            ************************************************************/
            #region Run Ordinal Matrix Factorization with NMF as scorer
            Utils.PrintHeading("Train NMF as scorer for OMF");
            if (Utils.Ask())
            {
                // Get ratings from scorer, for both train and test
                // R_all contains indexes of all ratings both train and test
                RatingMatrix R_all = new RatingMatrix(R_unknown.UserCount, R_unknown.ItemCount);
                R_all.MergeNonOverlap(R_unknown);
                R_all.MergeNonOverlap(R_train.IndexesOfNonZeroElements());

                RatingMatrix R_predictedByNMF = NMF.PredictRatings(R_train, R_all, Config.NMF.MaxEpoch,
                    Config.NMF.LearnRate, Config.NMF.Regularization, Config.NMF.K);

                // Prediction
                Utils.PrintHeading("Ordinal Matrix Factorization with NMF as scorer");
                Utils.StartTimer();
                RatingMatrix R_predicted = new RatingMatrix(
                    OMF.PredictRatings(R_train.Matrix, R_unknown.Matrix, R_predictedByNMF.Matrix)
                    );
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

                //Utils.Pause();
            }
            #endregion


            /************************************************************
             *   Preferecen relations based Non-negative Matrix Factorization
            ************************************************************/
            #region Run preferecen relations based PrefNMF
            Utils.PrintHeading("Preferecen relations based PrefNMF");
            if (Utils.Ask())
            {
                // Prediction
                Utils.StartTimer();
                // PR_test should be replaced with PR_unknown, but for now it is the same
                PrefRelations PR_predicted = PrefNMF.PredictPrefRelations(PR_train, PR_test, Config.PrefNMF.MaxEpoch, Config.PrefNMF.LearnRate,  Config.PrefNMF.RegularizationOfUser, Config.PrefNMF.RegularizationOfItem, Config.PrefNMF.K);
                RatingMatrix R_predicted = new RatingMatrix(PR_predicted.GetPositionMatrix());
                //R_predicted.NormalizeInplace(-1, 1, 0, 1);
                
                //RatingMatrix R_predicted = PrefNMF.PredictRatings(PR_train, R_unknown, Config.PrefNMF.MaxEpoch,Config.PrefNMF.LearnRate, Config.PrefNMF.RegularizationOfUser,Config.PrefNMF.RegularizationOfItem, Config.PrefNMF.K);
                Utils.StopTimer();

                // Evaluation
                //R_predicted.Matrix.MapInplace(x => RecSys.Core.SpecialFunctions.InverseLogit(x), Zeros.AllowSkip);
                Utils.WriteMatrix(R_predicted.Matrix, "debug.csv");
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, Config.TopN);
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("Precision@" + n, Precision.Evaluate(relevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }
            }
            #endregion


            /************************************************************
             *   Rating based UserKNN
            ************************************************************/
            #region Run rating based UserKNN
            Utils.PrintHeading("Rating based User KNN");
            if (Utils.Ask())
            {
                // Prediction
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

                //Utils.Pause();
            }
            #endregion



            /************************************************************
             *   Preference relation based UserKNN
            ************************************************************/
            #region Run preference relation based UserKNN
            if (Utils.Ask())
            {
                // Prediction
                Utils.PrintHeading("Preference relation based UserKNN");
                Utils.StartTimer();
                RatingMatrix R_predicted = PrefUserKNN.PredictRatings(PR_train, R_unknown, Config.KNN.K);
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

        }
    }
}
