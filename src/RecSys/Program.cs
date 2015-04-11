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

namespace RecSys
{
    class Program
    {
        static void Main(string[] args)
        {
            #region Prepare rating data
            Utils.PrintHeading("Prepare rating data");
            Utils.StartTimer();
            RatingMatrix R_train;
            RatingMatrix R_test;
            //Utilities.LoadMovieLens("u.data", out R_test, out R_train, 60, 10);   // This will produce the normal performance of NMF, KNN, etc.
            Utils.LoadMovieLensSplitByCount("u.data", out R_train, out R_test);
            RatingMatrix R_unknown = R_test.IndexesOfNonZeroElements();
            Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utils.StopTimer());
            Console.WriteLine(R_train.DatasetBrief("Train set"));
            Console.WriteLine(R_test.DatasetBrief("Test set"));
            Utils.WriteMatrix(R_train.Matrix, "R_train.csv");
            Utils.WriteMatrix(R_test.Matrix, "R_test.csv");

            Console.WriteLine("Extract relevant items from test set with threshold = " 
                + Config.Ratings.RelevanceThreshold);
            Dictionary<int, List<int>> relevantItemsByUser =
                ItemRecommendationCore.GetRelevantItemsByUser(R_test, Config.Ratings.RelevanceThreshold);
            Console.WriteLine("Average # of relevant items per user = "
                + relevantItemsByUser.Average(k => k.Value.Count).ToString("0.0"));
            Utils.Pause();
            #endregion

            #region Run Global Mean
            if (Config.RunGlobalMean)
            {
                Utils.PrintHeading("Global Mean");
                double globalMean = R_train.GetGlobalMean();
                RatingMatrix R_predicted = R_unknown.Multiply(globalMean);
                Console.WriteLine("{0,-23} │ {1,13:0.0000}" , 
                    "RMSE", RMSE.Evaluate(R_test, R_predicted));
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "MAE", MAE.Evaluate(R_test, R_predicted));
            }
            Utils.Pause();
            #endregion

            #region Run rating based UserKNN
            if (Config.RunRatingUserKNN)
            {
                Utils.PrintHeading("Rating based User KNN");

                // Compute or load similarities
                if (Config.RecomputeSimilarity)
                {
                    Console.WriteLine("Compute user-user similarities ... ");
                    Utils.StartTimer();
                    R_train.UserSimilarities = R_train.UserPearson();
                    Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utils.StopTimer());
                    Utils.WriteMatrix(R_train.UserSimilarities, Config.Ratings.UserSimilarityFile);

                    Console.WriteLine("Total similarities=" + R_train.UserSimilarities.RowSums().Sum().ToString());
                    Console.WriteLine("Total abs similarities=" + R_train.UserSimilarities.RowAbsoluteSums().Sum().ToString());
                }
                else
                {
                    Console.Write("Load user-user similarities ... ");
                    R_train.UserSimilarities = Utils.ReadDenseMatrix(Config.Ratings.UserSimilarityFile);
                    Console.WriteLine("completed.");
                }

                // Prediction
                Utils.StartTimer();
                RatingMatrix R_predicted = UserKNN.PredictRatings(R_train, R_unknown, Config.KNN.K);
                Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utils.StopTimer());

                // Evaluation
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "RMSE", RMSE.Evaluate(R_test, R_predicted));
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "MAE", MAE.Evaluate(R_test, R_predicted));
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}",
                    "NCDG@" + i, NCDG.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}",
                    "Precision@" + i, Precision.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }
                
                Utils.Pause();
            }
            #endregion

            #region Run rating based NMF
            if (Config.RunNMF)
            {
                Utils.PrintHeading("Rating based NMF");
                
                // Prediction
                Utils.StartTimer();
                RatingMatrix R_predicted = NMF.PredictRatings(R_train, R_unknown, Config.NMF.MaxEpoch,Config.NMF.LearnRate, Config.NMF.Regularization, Config.NMF.K);
                Console.WriteLine("{0,-23} │ {1,18:0.000}s", "Computation time", Utils.StopTimer());
                
                // Evaluation
                Console.WriteLine("{0,-23} │ {1,19:0.0000}" ,"RMSE", RMSE.Evaluate(R_test, R_predicted));
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "MAE", MAE.Evaluate(R_test, R_predicted));
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}", "NCDG@" + i, NCDG.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}", "Precision@" + i, Precision.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }
                Utils.Pause();
            }
            #endregion

            #region Prepare preference relation data
            Utils.PrintHeading("Prepare preferecen relation data");
            Utils.StartTimer();
            PreferenceRelations PR_train = PreferenceRelations.CreateDiscrete(R_train);
            PreferenceRelations PR_test = PreferenceRelations.CreateDiscrete(R_test);
            Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utils.StopTimer());
            List<int> targetUsers = PR_test.Users;

            // Note that both preference-based and rating-based methods will be checked against
            // the relevant items extracted from RATING data sets.
            // The correct order of recommended items by user
            //Dictionary<int, List<int>> userRecommendations_truth = new Dictionary<int, List<int>>(targetUsers.Count);
            //Utilities.StartTimer("Generate truth recommendation lists from PR ... \n");
            //userRecommendations_truth = PR_test.GetRecommendations(Config.TopN);
            //Console.WriteLine("completed in {0:0.000}s", Utilities.StopTimer());

            #endregion



            /*

            SparseMatrix positionMatrix = PR_train.GetPositionMatrix();
            RatingMatrix tempMatrix = new RatingMatrix(positionMatrix);
            RatingMatrix myRatingMatrixFromPositions = new RatingMatrix(
            positionMatrix.Multiply(2.5).Add(tempMatrix.IndexOfNonZeroElements().Matrix.Multiply(2.5))
            );
            Console.WriteLine(myRatingMatrixFromPositions.DatasetBrief("Rating matrix from positions"));
            for (int user = 0; user < R_train.UserCount; user++ )
            {
                for(int item = 0 ; item < R_train.ItemCount; item++)
                {
                    if (positionMatrix[user, item] == 0 && R_train[user, item] != 0)
                    {
                        Console.WriteLine("=====positionMatrix[{0},{1}] == {2}", user, item, positionMatrix[user, item]);
                        Console.WriteLine("=====R_train[{0},{1}] == {2}", user, item, R_train[user, item]);
                    }

                    if (R_train[user, item] == 0 && positionMatrix[user, item] != 0)
                    {
                        Console.WriteLine("******positionMatrix[{0},{1}] == {2}", user, item, positionMatrix[user, item]);
                        Console.WriteLine("******R_train[{0},{1}] == {2}", user, item, R_train[user, item]);
                    }
                }
            }


            // Prediction
            myRatingMatrixFromPositions.UserSimilarities = Utilities.ReadDenseMatrix(Config.Ratings.UserSimilarityFile);
            Utilities.StartTimer();
            RatingMatrix R_predicted2 = UserKNN.PredictRatings(myRatingMatrixFromPositions, 
                R_unknown, Config.KNN.K);
            Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utilities.StopTimer());

            // Evaluation
            Console.WriteLine("{0,-23} │ {1,13:0.0000}", "RMSE", RMSE.Evaluate(R_test, R_predicted2));
            Console.WriteLine("{0,-23} │ {1,13:0.0000}", "MAE", MAE.Evaluate(R_test, R_predicted2));
            for (int i = 1; i <= 10; i++)
            {
                Console.WriteLine("{0,-23} │ {1,13:0.0000}",
                "NCDG@" + i, NCDG.Evaluate(relevantItemsByUser, R_predicted2.GetUserTopNItems(i), i));
            }

            Utilities.Pause();

            */











            #region Run preference relation based UserKNN
            if (Config.RunPreferenceUserKNN)
            {
                // Compute or load similarities
                if (true)
                {
                    Utils.PrintHeading("Compute Preference Relation based similarities");
                    Utils.StartTimer();
                    //PR_train.UserSimilarities = Utilities.ReadDenseMatrix(Config.Ratings.UserSimilarityFile); //TODO: PR_train.UserCosine();
                    PR_train.UserSimilarities = PR_train.UserCosine();
                    Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utils.StopTimer());
                    Utils.WriteMatrix(PR_train.UserSimilarities, "userSimilaritiesPR.csv");

                    Console.WriteLine("Total similarities = " + PR_train.UserSimilarities.RowSums().Sum().ToString("0.0"));
                    Console.WriteLine("Total abs similarities = " + PR_train.UserSimilarities.RowAbsoluteSums().Sum().ToString("0.0"));
                    Utils.PrintHeading("Preference Relation based User KNN");
                }
                else
                {
                    Utils.PrintHeading("Preference Relation based User KNN");
                    Console.Write("Load user-user similarities ... ");
                    PR_train.UserSimilarities = Utils.ReadDenseMatrix("userSimilaritiesPR.csv");
                    Console.WriteLine("completed.");
                }
                
                // Prediction
                Utils.StartTimer();
                RatingMatrix PR_predicted = PrefUserKNN.PredictRatings(PR_train, R_unknown, Config.KNN.K);
               Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utils.StopTimer());

               // Evaluation
               for (int i = 1; i <= 10; i++)
               {
                   Console.WriteLine("{0,-23} │ {1,13:0.0000}", "NCDG@" + i,
                       NCDG.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(PR_predicted, i), i));
               }

               // Evaluation
               for (int i = 1; i <= 10; i++)
               {
                   Console.WriteLine("{0,-23} │ {1,13:0.0000}", "Precision@" + i,
                       Precision.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(PR_predicted, i), i));
               }
                //Console.WriteLine("{0,-23} │ {1,13:0.0000}" ,
                //   "Precision@" + Config.TopN,
                //   RecSys.MAP.AveragePrecisionAtN(relevantItemsByUser, userRecommendations_predicted, Config.TopN));
                Utils.Pause();
            }
            #endregion


            #region Run rating based UserKNN with PR similarities
            if (Config.RunRatingUserKNN)
            {
                Utils.PrintHeading("Rating based User KNN with PR similarities");
                    R_train.UserSimilarities = Utils.ReadDenseMatrix("userSimilaritiesPR.csv");

                // Prediction
                Utils.StartTimer();
                RatingMatrix R_predicted = UserKNN.PredictRatings(R_train, R_unknown, Config.KNN.K);
                Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utils.StopTimer());

                // Evaluation
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "RMSE", RMSE.Evaluate(R_test, R_predicted));
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "MAE", MAE.Evaluate(R_test, R_predicted));
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}",
                    "NCDG@" + i, NCDG.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}",
                    "Precision@" + i, Precision.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }

                Utils.Pause();
            }
            #endregion

            #region Run rating based UserKNN with PR similarities and positions
            if (Config.RunRatingUserKNN)
            {
                Utils.PrintHeading("Rating based User KNN with PR similarities and positions");

                SparseMatrix positionMatrix = PR_train.GetPositionMatrix();
                RatingMatrix tempMatrix = new RatingMatrix(positionMatrix);
                RatingMatrix ratingMatrixFromPositions = new RatingMatrix(
                positionMatrix.Multiply(2.5).Add(tempMatrix.IndexesOfNonZeroElements().Matrix.Multiply(2.5))
                );
                Console.WriteLine(ratingMatrixFromPositions.DatasetBrief("Rating matrix from positions"));

                ratingMatrixFromPositions.UserSimilarities = Utils.ReadDenseMatrix("userSimilaritiesPR.csv");

                // Prediction
                Utils.StartTimer();
                RatingMatrix R_predicted = UserKNN.PredictRatings(ratingMatrixFromPositions, R_unknown, Config.KNN.K);
                Console.WriteLine("{0,-23} │ {1,12:0.000}s", "Computation time", Utils.StopTimer());

                // Evaluation
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "RMSE", RMSE.Evaluate(R_test, R_predicted));
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "MAE", MAE.Evaluate(R_test, R_predicted));
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}",
                    "NCDG@" + i, NCDG.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}",
                    "Precision@" + i, Precision.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }

                Utils.Pause();
            }
            #endregion

            #region Run rating based NMF with positions
            if (Config.RunNMF)
            {
                Utils.PrintHeading("Rating based NMF with positions");

                SparseMatrix positionMatrix = PR_train.GetPositionMatrix();
                RatingMatrix tempMatrix = new RatingMatrix(positionMatrix);
                RatingMatrix ratingMatrixFromPositions = new RatingMatrix(
                positionMatrix.Multiply(2.5).Add(tempMatrix.IndexesOfNonZeroElements().Matrix.Multiply(2.5))
                );
                Console.WriteLine(ratingMatrixFromPositions.DatasetBrief("Rating matrix from positions"));

                // Prediction
                Utils.StartTimer();
                RatingMatrix R_predicted = NMF.PredictRatings(ratingMatrixFromPositions, R_unknown, Config.NMF.MaxEpoch, Config.NMF.LearnRate, Config.NMF.Regularization, Config.NMF.K);
                Console.WriteLine("{0,-23} │ {1,18:0.000}s", "Computation time", Utils.StopTimer());

                // Evaluation
                Console.WriteLine("{0,-23} │ {1,19:0.0000}", "RMSE", RMSE.Evaluate(R_test, R_predicted));
                Console.WriteLine("{0,-23} │ {1,13:0.0000}", "MAE", MAE.Evaluate(R_test, R_predicted));
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}", "NCDG@" + i, NCDG.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }
                for (int i = 1; i <= 10; i++)
                {
                    Console.WriteLine("{0,-23} │ {1,13:0.0000}", "Precision@" + i, Precision.Evaluate(relevantItemsByUser, ItemRecommendationCore.GetTopNItemsByUser(R_predicted, i), i));
                }
                Utils.Pause();
            }
            #endregion

        }
    }
}
