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
             *      itemSimilaritiesOfRating    => The item-item similarities from R_train
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
            Console.WriteLine("Converting R_train into PR_train");
            PrefRelations PR_train = PrefRelations.CreateDiscrete(R_train);
            Console.WriteLine("Converting R_test into PR_test");
            PrefRelations PR_test = PrefRelations.CreateDiscrete(R_test);
            //PrefRelations PR_train = PrefRelations.CreateScalar(R_train);
            Utils.StopTimer();
            #endregion

            #region Compute or load similarities
            Matrix<double> userSimilaritiesOfRating;
            Matrix<double> userSimilaritiesOfPref;
            Matrix<double> itemSimilaritiesOfRating;
            //Matrix<double> itemSimilaritiesOfPref;
            if (Config.LoadSavedData)
            {
                Utils.StartTimer();
                Utils.PrintHeading("Load user-user similarities from R_train");
                userSimilaritiesOfRating = Utils.ReadDenseMatrix(Config.UserSimilaritiesOfRatingFile);
                Utils.PrintValue("Sum of similarities", userSimilaritiesOfRating.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", userSimilaritiesOfRating.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Load item-item similarities from R_train");
                itemSimilaritiesOfRating = Utils.ReadDenseMatrix(Config.ItemSimilaritiesOfRatingFile);
                Utils.PrintValue("Sum of similarities", itemSimilaritiesOfRating.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", itemSimilaritiesOfRating.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Load user-user similarities from PR_train");
                userSimilaritiesOfPref = Utils.ReadDenseMatrix(Config.UserSimilaritiesOfPrefFile);
                Utils.PrintValue("Sum of similarities", userSimilaritiesOfPref.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", userSimilaritiesOfPref.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();

                // TODO: add PR based item-item similarities
            }
            else
            {
                Utils.StartTimer();
                Utils.PrintHeading("Compute user-user similarities from R_train");
                userSimilaritiesOfRating = Metric.GetPearsonOfRows(R_train);
                Utils.WriteMatrix(userSimilaritiesOfRating, Config.UserSimilaritiesOfRatingFile);
                Utils.PrintValue("Sum of similarities", userSimilaritiesOfRating.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", userSimilaritiesOfRating.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Compute item-item similarities from R_train");
                itemSimilaritiesOfRating = Metric.GetPearsonOfColumns(R_train);
                Utils.WriteMatrix(itemSimilaritiesOfRating, Config.ItemSimilaritiesOfRatingFile);
                Vector<double> rowSums = itemSimilaritiesOfRating.RowSums();
                double sum = checked(rowSums.Sum());
                Utils.PrintValue("Sum of similarities", itemSimilaritiesOfRating.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", itemSimilaritiesOfRating.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Compute user-user similarities from PR_train");
                userSimilaritiesOfPref = Metric.GetCosineOfPrefRelations(PR_train);
                Utils.WriteMatrix(userSimilaritiesOfPref, Config.UserSimilaritiesOfPrefFile);
                Utils.PrintValue("Sum of similarities", userSimilaritiesOfPref.RowSums().Sum().ToString("0.0000"));
                Utils.PrintValue("Abs sum of similarities", userSimilaritiesOfPref.RowAbsoluteSums().Sum().ToString("0.0000"));
                Utils.StopTimer();
            }

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
             *   Ordinal Matrix Factorization with PrefNMF as scorer
            ************************************************************/
            #region Run Ordinal Matrix Factorization with PrefNMF as scorer
            Utils.PrintHeading("Train PrefNMF as scorer for OMF");
            if (Utils.Ask())
            {
                // Get ratings from scorer, for both train and test
                // R_all contains indexes of all ratings both train and test
                RatingMatrix R_all = new RatingMatrix(R_unknown.UserCount, R_unknown.ItemCount);
                R_all.MergeNonOverlap(R_unknown);
                R_all.MergeNonOverlap(R_train.IndexesOfNonZeroElements());
                PrefRelations PR_unknown = PrefRelations.CreateDiscrete(R_all);

                // Prediction
                Utils.StartTimer();
                // PR_test should be replaced with PR_unknown, but for now it is the same
                PrefRelations PR_predicted = PrefNMF.PredictPrefRelations(PR_train, PR_unknown, Config.PrefNMF.MaxEpoch, Config.PrefNMF.LearnRate, Config.PrefNMF.RegularizationOfUser, Config.PrefNMF.RegularizationOfItem, Config.PrefNMF.K);

                // Both predicted and train need to be quantized
                // otherwise OMF won't accept
                PR_predicted.quantization(0, 1.0, new List<double> { Config.Preferences.LessPreferred, Config.Preferences.EquallyPreferred, Config.Preferences.Preferred });
                RatingMatrix R_predictedByPrefNMF = new RatingMatrix(PR_predicted.GetPositionMatrix());

                // PR_train itself is already in quantized form!
                //PR_train.quantization(0, 1.0, new List<double> { Config.Preferences.LessPreferred, Config.Preferences.EquallyPreferred, Config.Preferences.Preferred });
                RatingMatrix R_train_positions = new RatingMatrix(PR_train.GetPositionMatrix());
                R_train_positions.Quantization(1, 2, new List<double> { 1, 2, 3 });
                Utils.StopTimer();

                Console.WriteLine();

                // Prediction
                Utils.PrintHeading("Ordinal Matrix Factorization with PrefNMF as scorer");
                Utils.StartTimer();
                RatingMatrix R_predicted = new RatingMatrix(OMF.PredictRatings(R_train_positions.Matrix, R_unknown.Matrix, R_predictedByPrefNMF.Matrix, Config.Preferences.quantizerThree));
                Utils.StopTimer();

                // Prediction
                Utils.PrintHeading("Generate ordinal distributinos on R_all (train+test) for ORF");
                Utils.StartTimer();
                OMF.PredictRatings(R_train_positions.Matrix, R_all.Matrix, R_predictedByPrefNMF.Matrix, Config.Preferences.quantizerThree, "OMFDistribution.txt");
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

                //Utils.Pause();
            }
            #endregion


            /************************************************************
             *   ORF+OMF(PrefNMF)
            ************************************************************/
            #region ORF+OMF(PrefNMF)
            Utils.PrintHeading("ORF+OMF(PrefNMF)");
            if (Utils.Ask())
            {
                RatingMatrix R_predicted_expectations;
                RatingMatrix R_predicted_mostlikely;
                Dictionary<Tuple<int,int>,double[]> OMFDistributions = Utils.LoadOMFDistributions("probabilities.txt");
                ORF orf = new ORF();
                // Prediction
                Utils.StartTimer();
                orf.PredictRatings(
                    R_train,R_unknown,itemSimilaritiesOfRating,OMFDistributions,1,0.001,0.1,1000,3, out R_predicted_expectations, out R_predicted_mostlikely);
                Utils.StopTimer();

                // Evaluation
                var topNItemsByUser_expectations = ItemRecommendationCore.GetTopNItemsByUser(R_predicted_expectations, Config.TopN);
                var topNItemsByUser_mostlikely = ItemRecommendationCore.GetTopNItemsByUser(R_predicted_mostlikely, Config.TopN);
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(relevantItemsByUser, topNItemsByUser_expectations, n).ToString("0.0000"));
                }
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("Precision@" + n, Precision.Evaluate(relevantItemsByUser, topNItemsByUser_expectations, n).ToString("0.0000"));
                }
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(relevantItemsByUser, topNItemsByUser_mostlikely, n).ToString("0.0000"));
                }
                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("Precision@" + n, Precision.Evaluate(relevantItemsByUser, topNItemsByUser_mostlikely, n).ToString("0.0000"));
                }
                //Utils.Pause();
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
                    OMF.PredictRatings(R_train.Matrix, R_unknown.Matrix, R_predictedByNMF.Matrix, 
            Config.Preferences.quantizerFive)
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
             *   Orginal Preferecen relations based Non-negative Matrix Factorization
            ************************************************************/
            #region Run preferecen relations based PrefNMF
            Utils.PrintHeading("Orginal Preferecen relations based PrefNMF");
            if (Utils.Ask())
            {
                // Prediction
                Utils.StartTimer();
                RatingMatrix R_predicted = PrefNMF.PredictRatings(PR_train, R_unknown, Config.PrefNMF.MaxEpoch, Config.PrefNMF.LearnRate, Config.PrefNMF.RegularizationOfUser, Config.PrefNMF.RegularizationOfItem, Config.PrefNMF.K);
                Utils.StopTimer();

                // Evaluation
                //Utils.WriteMatrix(R_predicted.Matrix, "debug.csv");
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
             *   Positions based Preferecen relations based Non-negative Matrix Factorization
            ************************************************************/
            #region Run preferecen relations based PrefNMF
            Utils.PrintHeading("Positions based Preferecen relations based PrefNMF");
            if (Utils.Ask())
            {
                // Prediction
                Utils.StartTimer();
                // PR_test should be replaced with PR_unknown, but for now it is the same
                PrefRelations PR_predicted = PrefNMF.PredictPrefRelations(PR_train, PR_test, Config.PrefNMF.MaxEpoch, Config.PrefNMF.LearnRate,  Config.PrefNMF.RegularizationOfUser, Config.PrefNMF.RegularizationOfItem, Config.PrefNMF.K);
                PR_predicted.quantization(0, 1.0, new List<double> { Config.Preferences.LessPreferred, Config.Preferences.EquallyPreferred, Config.Preferences.Preferred });
                RatingMatrix R_predicted = new RatingMatrix(PR_predicted.GetPositionMatrix());
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
                RatingMatrix R_predicted = UserKNN.PredictRatings(R_train, R_unknown, userSimilaritiesOfRating, Config.KNN.K);
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
