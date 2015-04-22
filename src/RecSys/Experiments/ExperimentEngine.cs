using MathNet.Numerics.LinearAlgebra;
using RecSys.Core;
using RecSys.Evaluation;
using RecSys.Numerical;
using RecSys.Ordinal;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace RecSys.Experiments
{
    [Serializable]
    public class ExperimentEngine
    {
        /************************************************************
         *   R_train     => Rating Matrix train set
         *   R_test      => Rating Matrix test set
         *   R_unknown   => Rating Matrix with ones indicating unknown entries in the R_test
         *   PR_train    => Preference relations constructed from R_train
         *   PR_test     => Preference relations constructed from R_test
         *   UserSimilaritiesOfRating    => The user-user similarities from R_train
         *   ItemSimilaritiesOfRating    => The item-item similarities from R_train
         *   UserSimilaritiesOfPref      => The user-user similarities from PR_train
         *   ItemSimilaritiesOfPref      => The user-user similarities from PR_train
         *   RelevantItemsByUser         => The relevant items of each user based on R_test, 
         *                                  is used as ground truth in all ranking evalution
        ************************************************************/

        #region Experiment settings
        public RatingMatrix R_train;
        public RatingMatrix R_test;
        public RatingMatrix R_unknown;
        public PrefRelations PR_train;
        public PrefRelations PR_test;
        public SimilarityData UserSimilaritiesOfRating;
        public SimilarityData UserSimilaritiesOfPref;
        public SimilarityData ItemSimilaritiesOfRating;
        public SimilarityData ItemSimilaritiesOfPref;
        public HashSet<Tuple<int, int>> StrongSimilarityIndicatorsByItemRating;
        public HashSet<Tuple<int, int>> StrongSimilarityIndicatorsByItemPref;
        public bool ReadyForNumerical;
        public bool ReadyForOrdinal;
        public string DataSetFile;
        public int MinCountOfRatings;
        public int CountOfRatingsForTrain;
        public bool ShuffleData;
        public int Seed;
        public double RelevantItemCriteria;
        public int MaxCountOfNeighbors;
        public double StrongSimilarityThreshold;
        public Dictionary<int, List<int>> RelevantItemsByUser;
        #endregion

        #region Constructor
        public ExperimentEngine(string dataSetFile, int minCountOfRatings,
            int countOfRatingsForTrain, bool shuffleData, int seed, double relevantItemCriteria,
            int maxCountOfNeighbors, double strongSimilarityThreshold)
        {
            DataSetFile = dataSetFile;
            MinCountOfRatings = minCountOfRatings;
            CountOfRatingsForTrain = countOfRatingsForTrain;
            ShuffleData = shuffleData;
            Seed = seed;
            RelevantItemCriteria = relevantItemCriteria;
            MaxCountOfNeighbors = maxCountOfNeighbors;
            ReadyForNumerical = false;
            ReadyForOrdinal = false;
            StrongSimilarityThreshold = strongSimilarityThreshold;
        }
        public ExperimentEngine() { }
        #endregion

        #region GetDataFileName
        // When save to data to file, the settings will be encoded into file name
        private string GetDataFileName()
        {
            string output = "";
            output+= DataSetFile;
            output+= "_S" + Seed;
            output+= "_MCR" + MinCountOfRatings;
            output += "_CRT" + CountOfRatingsForTrain;
            output += "_MCN" + MaxCountOfNeighbors;
            output += "_SST" + StrongSimilarityThreshold.ToString("0.00");
            output +=".bin";

            return output;
        }
        #endregion

        #region Get ready for numerical methods
        public string GetReadyForNumerical(bool saveLoadedData = true)
        {
            if (ReadyForNumerical) { return "Is ready."; }

            StringBuilder log = new StringBuilder();
            Utils.StartTimer();

            log.AppendLine(Utils.PrintHeading("Create R_train/R_test sets from " + DataSetFile));
            Utils.LoadMovieLensSplitByCount(DataSetFile, out R_train,
                out R_test, MinCountOfRatings, CountOfRatingsForTrain, ShuffleData, Seed);

            Console.WriteLine(R_train.DatasetBrief("Train set"));
            Console.WriteLine(R_test.DatasetBrief("Test set"));
            log.AppendLine(R_train.DatasetBrief("Train set"));
            log.AppendLine(R_test.DatasetBrief("Test set"));

            R_unknown = R_test.IndexesOfNonZeroElements();

            log.AppendLine(Utils.PrintValue("Relevant item criteria", RelevantItemCriteria.ToString("0.0")));
            RelevantItemsByUser = ItemRecommendationCore.GetRelevantItemsByUser(R_test, RelevantItemCriteria);
            log.AppendLine(Utils.PrintValue("Mean # of relevant items per user",
                RelevantItemsByUser.Average(k => k.Value.Count).ToString("0")));
            log.AppendLine(Utils.StopTimer());

            #region Prepare similarity data
            if (File.Exists("USR_" + GetDataFileName())
                && File.Exists("ISR_" + GetDataFileName())
                && File.Exists("SSIIR_" + GetDataFileName()))
            {
                Utils.StartTimer();
                Utils.PrintHeading("Load user-user similarities (rating based)");
                UserSimilaritiesOfRating = Utils.IO<SimilarityData>.LoadObject("USR_" + GetDataFileName());
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Load item-item similarities (rating based)");
                ItemSimilaritiesOfRating = Utils.IO<SimilarityData>.LoadObject("ISR_" + GetDataFileName());
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Load item-item strong similarity indicators (rating based)");
                StrongSimilarityIndicatorsByItemRating = Utils.IO<HashSet<Tuple<int, int>>>.LoadObject("SSIIR_" + GetDataFileName());
                Utils.StopTimer();
            }
            else
            {
                Utils.StartTimer();
                Utils.PrintHeading("Compute user-user similarities (rating based)");
                Metric.GetCosineOfRows(R_train, MaxCountOfNeighbors,StrongSimilarityThreshold,
                    out UserSimilaritiesOfRating);
                if (saveLoadedData) 
                {
                    Utils.IO<SimilarityData>.SaveObject(UserSimilaritiesOfRating, "USR_" + GetDataFileName());
                }
                Utils.StopTimer();

                Utils.StartTimer();
                Utils.PrintHeading("Compute item-item similarities (rating based)");
                Metric.GetCosineOfColumns(R_train, MaxCountOfNeighbors, StrongSimilarityThreshold, 
                    out ItemSimilaritiesOfRating, out StrongSimilarityIndicatorsByItemRating);
                if (saveLoadedData)
                {
                    Utils.IO<SimilarityData>.SaveObject(ItemSimilaritiesOfRating, "ISR_" + GetDataFileName());
                    Utils.IO<HashSet<Tuple<int,int>>>
                        .SaveObject(StrongSimilarityIndicatorsByItemRating, "SSIIR_" + GetDataFileName());
                }
                Utils.StopTimer();
            }
            #endregion

            ReadyForNumerical = true;

            return log.ToString();
        }
        #endregion

        #region Get ready for ordinal methods
        public string GetReadyForOrdinal(bool saveLoadedData = true)
        {
            if (!ReadyForNumerical) { GetReadyForNumerical(); }
            if (ReadyForOrdinal) { return "Is ready."; }

            StringBuilder log = new StringBuilder();
            Utils.StartTimer();
            log.AppendLine(Utils.PrintHeading("Prepare preferecen relation data"));

            Console.WriteLine("Converting R_train into PR_train");
            log.AppendLine("Converting R_train into PR_train");
            PR_train = PrefRelations.CreateDiscrete(R_train);

            Console.WriteLine("Converting R_test into PR_test");
            log.AppendLine("Converting R_test into PR_test");
            PR_test = PrefRelations.CreateDiscrete(R_test);

            log.AppendLine(Utils.StopTimer());

            #region Prepare similarity data
            if (File.Exists("USP_" + GetDataFileName())
                && File.Exists("ISP_" + GetDataFileName())
                && File.Exists("SSIIP_" + GetDataFileName()))
            {

                Utils.StartTimer();
                Utils.PrintHeading("Load user, item, indicators variables (Pref based)");
                UserSimilaritiesOfPref = Utils.IO<SimilarityData>.LoadObject("USP_" + GetDataFileName());
                ItemSimilaritiesOfPref = Utils.IO<SimilarityData>.LoadObject("ISP_" + GetDataFileName());
                StrongSimilarityIndicatorsByItemPref = Utils.IO<HashSet<Tuple<int,int>>>.LoadObject("SSIIP_" + GetDataFileName());
                Utils.StopTimer();
            }
            else
            {
                Utils.StartTimer();
                Utils.PrintHeading("Compute user-user similarities (Pref based)");
                Metric.GetCosineOfPrefRelations(PR_train, MaxCountOfNeighbors, 
                    StrongSimilarityThreshold, out UserSimilaritiesOfPref);
                Utils.StopTimer();

                // For the moment, we use user-wise preferences to compute
                // item-item similarities, it is not the same as user-user pref similarities
                Utils.StartTimer();
                Utils.PrintHeading("Compute item-item similarities (Pref based)");
                RatingMatrix PR_userwise_preferences = new RatingMatrix(PR_train.GetPositionMatrix());
                Metric.GetCosineOfColumns(PR_userwise_preferences, MaxCountOfNeighbors, StrongSimilarityThreshold,
                    out ItemSimilaritiesOfPref, out StrongSimilarityIndicatorsByItemPref);
                Utils.StopTimer();

                if (saveLoadedData)
                {
                    Utils.IO<SimilarityData>.SaveObject(UserSimilaritiesOfPref, "USP_" + GetDataFileName());
                    Utils.IO<SimilarityData>.SaveObject(ItemSimilaritiesOfPref, "ISP_" + GetDataFileName());
                    Utils.IO<HashSet<Tuple<int,int>>>
                        .SaveObject(StrongSimilarityIndicatorsByItemPref, "SSIIP_" + GetDataFileName());
                }
                Utils.StopTimer();

            }
            #endregion

            

            ReadyForOrdinal = true;

            return log.ToString();
        }
        #endregion

        #region Get ready for all methods
        public string GetReadyAll()
        {
            StringBuilder log = new StringBuilder();
            if(!ReadyForNumerical)
                log.AppendLine(GetReadyForNumerical());
            if(!ReadyForOrdinal)
                log.AppendLine(GetReadyForOrdinal());

            return log.ToString();
        }
        #endregion

        #region Global Mean
        /// <summary>
        /// Predict all unknown values as global mean rating.
        /// </summary>
        public string RunGlobalMean()
        {
            if (!ReadyForNumerical) { GetReadyForNumerical(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("Global Mean"));

            // Prediction
            Utils.StartTimer();
            double globalMean = R_train.GetGlobalMean();
            RatingMatrix R_predicted = R_unknown.Multiply(globalMean);
            log.AppendLine(Utils.StopTimer());

            // Numerical Evaluation
            log.AppendLine(Utils.PrintValue("RMSE", RMSE.Evaluate(R_test, R_predicted).ToString("0.0000")));
            log.AppendLine(Utils.PrintValue("MAE", MAE.Evaluate(R_test, R_predicted).ToString("0.0000")));

            return log.ToString();
        }
        #endregion

        #region Most Popular
        /// <summary>
        /// Recommend the most popular (measured by mean rating) items to all users.
        /// </summary>
        public string RunMostPopular(int topN)
        {
            if (!ReadyForNumerical) { GetReadyForNumerical(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("Most popular"));

            // Prediction
            Utils.StartTimer();
            var meanByItem = R_train.GetItemMeans();
            RatingMatrix R_predicted = new RatingMatrix(R_unknown.UserCount, R_unknown.ItemCount);
            foreach (var element in R_unknown.Matrix.EnumerateIndexed(Zeros.AllowSkip))
            {
                int indexOfUser = element.Item1;
                int indexOfItem = element.Item2;
                R_predicted[indexOfUser, indexOfItem] = meanByItem[indexOfItem];
            }
            var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, topN);
            log.AppendLine(Utils.StopTimer());

            // TopN Evaluation
            for (int n = 1; n <= topN; n++)
            {
                log.AppendLine(Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser, n).ToString("0.0000")));
            }

            return log.ToString();
        }
        #endregion

        #region NMF
        /// <summary>
        /// Rating based Non-negative Matrix Factorization
        /// </summary>
        public string RunNMF(int maxEpoch, double learnRate, double regularization,
            int factorCount, int topN = 0)
        {
            if (!ReadyForNumerical) { GetReadyForNumerical(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("NMF"));

            // Prediction
            Utils.StartTimer();
            RatingMatrix R_predicted = NMF.PredictRatings(R_train, R_unknown, maxEpoch,
                learnRate, regularization, factorCount);
            log.AppendLine(Utils.StopTimer());

            // Numerical Evaluation
            log.AppendLine(Utils.PrintValue("RMSE", RMSE.Evaluate(R_test, R_predicted).ToString("0.0000")));
            log.AppendLine(Utils.PrintValue("MAE", MAE.Evaluate(R_test, R_predicted).ToString("0.0000")));

            // TopN Evaluation
            if (topN != 0)
            {
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, topN);
                for (int n = 1; n <= topN; n++)
                {
                    log.AppendLine(Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser, n).ToString("0.0000")));
                }
            }

            return log.ToString();
        }
        #endregion

        #region UserKNN
        public string RunUserKNN(int neighborCount, int topN = 0)
        {
            if (!ReadyForNumerical) { GetReadyForNumerical(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("UserKNN"));

            // Prediction
            Utils.StartTimer();
            RatingMatrix R_predicted = Numerical.UserKNN.PredictRatings(R_train, R_unknown, UserSimilaritiesOfRating, neighborCount);
            log.AppendLine(Utils.StopTimer());

            // Numerical Evaluation
            log.AppendLine(Utils.PrintValue("RMSE", RMSE.Evaluate(R_test, R_predicted).ToString("0.0000")));
            log.AppendLine(Utils.PrintValue("MAE", MAE.Evaluate(R_test, R_predicted).ToString("0.0000")));

            // TopN Evaluation
            if (topN != 0)
            {
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, topN);
                for (int n = 1; n <= topN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
                }
            }

            return log.ToString();
        }
        #endregion

        #region PrefNMF
        public string RunPrefNMF(int maxEpoch, double learnRate, double regularizationOfUser,
            double regularizationOfItem, int factorCount, int topN = 10)
        {
            if (!ReadyForOrdinal) { GetReadyForOrdinal(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("PrefNMF"));

            // Prediction
            Utils.StartTimer();
            RatingMatrix R_predicted = PrefNMF.PredictRatings(PR_train, R_unknown,
                maxEpoch, learnRate, regularizationOfUser, regularizationOfItem, factorCount);
            log.AppendLine(Utils.StopTimer());

            // Evaluation
            var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, topN);
            for (int n = 1; n <= topN; n++)
            {
                Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
            }

            return log.ToString();
        }
        #endregion

        #region PrefKNN
        public string RunPrefKNN(int neighborCount, int topN = 10)
        {
            if (!ReadyForOrdinal) { GetReadyForOrdinal(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("PrefKNN"));

            // Prediction
            Utils.StartTimer();
            RatingMatrix R_predicted = PrefUserKNN.PredictRatings(PR_train, R_unknown, neighborCount, UserSimilaritiesOfPref);
            log.AppendLine(Utils.StopTimer());

            // TopN Evaluation
            var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, topN);
            for (int n = 1; n <= topN; n++)
            {
                Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser, n).ToString("0.0000"));
            }

            return log.ToString();
        }
        #endregion

        #region PrefMRF: PrefNMF based ORF
        public string RunPrefMRF(double regularization, double learnRate, double minSimilarity, int maxEpoch, List<double> quantizer,
            int topN = 10)
        {
            // Load OMFDistribution from file
            Dictionary<Tuple<int, int>, List<double>> OMFDistributionByUserItem;
            if (File.Exists("PrefOMF_" + GetDataFileName()))
            {
                OMFDistributionByUserItem = Utils.IO<Dictionary<Tuple<int, int>, List<double>>>.LoadObject("PrefOMF_" + GetDataFileName());
            }
            else
            {
                return "Abort, Run OMF first.";
            }

            if (!ReadyForOrdinal) { GetReadyForOrdinal(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("PrefMRF: PrefNMF based ORF"));

            // Prediction
            Utils.StartTimer();
            RatingMatrix R_predicted_expectations;
            RatingMatrix R_predicted_mostlikely;

            // Convert PR_train into user-wise preferences
            RatingMatrix R_train_positions = new RatingMatrix(PR_train.GetPositionMatrix());
            R_train_positions.Quantization(quantizer[0], quantizer[quantizer.Count - 1] - quantizer[0], quantizer);

            ORF orf = new ORF();
            orf.PredictRatings( R_train_positions, R_unknown, StrongSimilarityIndicatorsByItemPref,
                OMFDistributionByUserItem, regularization, learnRate, maxEpoch, 
                quantizer.Count, out R_predicted_expectations, out R_predicted_mostlikely);
          
            log.AppendLine(Utils.StopTimer());

            // Evaluation
            var topNItemsByUser_expectations = ItemRecommendationCore.GetTopNItemsByUser(R_predicted_expectations, topN);
            var topNItemsByUser_mostlikely = ItemRecommendationCore.GetTopNItemsByUser(R_predicted_mostlikely, topN);
            for (int n = 1; n <= topN; n++)
            {
                log.AppendLine(Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser,
                    topNItemsByUser_expectations, n).ToString("0.0000")));
            }
            for (int n = 1; n <= topN; n++)
            {
                log.AppendLine(Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser,
                    topNItemsByUser_mostlikely, n).ToString("0.0000")));
            }

            return log.ToString();
        }
        #endregion

        #region NMF based ORF
        public string RunNMFbasedORF(double regularization, double learnRate, 
            int maxEpoch, List<double> quantizer, int topN = 0)
        {
            // Load OMFDistribution from file
            Dictionary<Tuple<int, int>, List<double>> OMFDistributionByUserItem;
            if (File.Exists("RatingOMF_" + GetDataFileName()))
            {
                OMFDistributionByUserItem = Utils.IO<Dictionary<Tuple<int, int>, List<double>>>.LoadObject("RatingOMF_" + GetDataFileName());
            }
            else
            {
                return "Abort, Run OMF first.";
            }

            if (!ReadyForNumerical) { GetReadyForNumerical(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("NMF based ORF"));

            // Prediction
            Utils.StartTimer();
            RatingMatrix R_predicted_expectations;
            RatingMatrix R_predicted_mostlikely;
            ORF orf = new ORF();
            orf.PredictRatings( R_train, R_unknown, StrongSimilarityIndicatorsByItemRating, 
                OMFDistributionByUserItem, regularization, learnRate, maxEpoch, 
                quantizer.Count, out R_predicted_expectations, out R_predicted_mostlikely);
            log.AppendLine(Utils.StopTimer());

            // Numerical Evaluation
            log.AppendLine(Utils.PrintValue("RMSE", RMSE.Evaluate(R_test, R_predicted_expectations).ToString("0.0000")));
            log.AppendLine(Utils.PrintValue("MAE", RMSE.Evaluate(R_test, R_predicted_mostlikely).ToString("0.0000")));

            // Top-N Evaluation
            if (topN != 0)
            {
                var topNItemsByUser_expectations = ItemRecommendationCore.GetTopNItemsByUser(R_predicted_expectations, topN);
                var topNItemsByUser_mostlikely = ItemRecommendationCore.GetTopNItemsByUser(R_predicted_mostlikely, topN);
                for (int n = 1; n <= topN; n++)
                {
                    log.AppendLine(Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser_expectations, n).ToString("0.0000")));
                }
                for (int n = 1; n <= topN; n++)
                {
                    log.AppendLine(Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser_mostlikely, n).ToString("0.0000")));
                }
            }

            return log.ToString();
        }
        #endregion

        #region PrefNMF based OMF
        public string RunPrefNMFbasedOMF(int maxEpoch, double learnRate, double regularizationOfUser,
            double regularizationOfItem, int factorCount, List<double> quantizer, int topN)
        {
            if (!ReadyForOrdinal) { GetReadyForOrdinal(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("PrefNMF based OMF"));

            // =============PrefNMF prediction on Train+Unknown============
            // Get ratings from scorer, for both train and test
            // R_all contains indexes of all ratings both train and test
            RatingMatrix R_all = new RatingMatrix(R_unknown.UserCount, R_unknown.ItemCount);
            R_all.MergeNonOverlap(R_unknown);
            R_all.MergeNonOverlap(R_train.IndexesOfNonZeroElements());
            PrefRelations PR_unknown = PrefRelations.CreateDiscrete(R_all);

            Utils.StartTimer();
            // PR_test should be replaced with PR_unknown, but it is the same
            PrefRelations PR_predicted = PrefNMF.PredictPrefRelations(PR_train, PR_unknown,
                maxEpoch, learnRate, regularizationOfUser, regularizationOfItem, factorCount);

            // Both predicted and train need to be quantized
            // otherwise OMF won't accept
            PR_predicted.quantization(0, 1.0,
                new List<double> { Config.Preferences.LessPreferred, 
                        Config.Preferences.EquallyPreferred, Config.Preferences.Preferred });
            RatingMatrix R_predictedByPrefNMF = new RatingMatrix(PR_predicted.GetPositionMatrix());

            // PR_train itself is already in quantized form!
            //PR_train.quantization(0, 1.0, new List<double> { Config.Preferences.LessPreferred, Config.Preferences.EquallyPreferred, Config.Preferences.Preferred });
            RatingMatrix R_train_positions = new RatingMatrix(PR_train.GetPositionMatrix());
            R_train_positions.Quantization(quantizer[0], quantizer[quantizer.Count - 1] - quantizer[0], quantizer);
            log.AppendLine(Utils.StopTimer());

            // =============OMF prediction on Train+Unknown============
            log.AppendLine(Utils.PrintHeading("Ordinal Matrix Factorization with PrefNMF as scorer"));
            Utils.StartTimer();
            Dictionary<Tuple<int, int>, List<double>> OMFDistributionByUserItem;
            RatingMatrix R_predicted;
            log.AppendLine(OMF.PredictRatings(R_train_positions.Matrix, R_unknown.Matrix, R_predictedByPrefNMF.Matrix,
                quantizer, out R_predicted, out OMFDistributionByUserItem));
            log.AppendLine(Utils.StopTimer());

            // TopN Evaluation
            var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, topN);
            for (int n = 1; n <= topN; n++)
            {
                log.AppendLine(Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser, n).ToString("0.0000")));
            }

            // Save OMFDistribution to file
            if (!File.Exists("PrefOMF_" + GetDataFileName()))
            {
                Utils.IO<Dictionary<Tuple<int, int>, List<double>>>.SaveObject(OMFDistributionByUserItem, "PrefOMF_" + GetDataFileName());
            }

            return log.ToString();
        }
        #endregion

        #region NMF based OMF
        public string RunNMFbasedOMF(int maxEpoch, double learnRate, double regularization, int factorCount,
            List<double> quantizer, int topN = 0)
        {
            if (!ReadyForNumerical) { GetReadyForNumerical(); }
            StringBuilder log = new StringBuilder();
            log.AppendLine(Utils.PrintHeading("NMF based OMF"));

            // NMF Prediction
            // Get ratings from scorer, for both train and test
            // R_all contains indexes of all ratings both train and test
            RatingMatrix R_all = new RatingMatrix(R_unknown.UserCount, R_unknown.ItemCount);
            R_all.MergeNonOverlap(R_unknown);
            R_all.MergeNonOverlap(R_train.IndexesOfNonZeroElements());
            Utils.StartTimer();
            RatingMatrix R_predictedByNMF = NMF.PredictRatings(R_train, R_all, maxEpoch,
                learnRate, regularization, factorCount);
            log.AppendLine(Utils.StopTimer());

            // OMF Prediction
            log.AppendLine(Utils.PrintHeading("Ordinal Matrix Factorization with NMF as scorer"));
            Utils.StartTimer();
            Dictionary<Tuple<int, int>, List<double>> OMFDistributionByUserItem;
            RatingMatrix R_predicted;
            log.AppendLine(OMF.PredictRatings(R_train.Matrix, R_unknown.Matrix, R_predictedByNMF.Matrix,
                quantizer, out R_predicted, out OMFDistributionByUserItem));
            log.AppendLine(Utils.StopTimer());

            // Numerical Evaluation
            log.AppendLine(Utils.PrintValue("RMSE", RMSE.Evaluate(R_test, R_predicted).ToString("0.0000")));
            log.AppendLine(Utils.PrintValue("MAE", MAE.Evaluate(R_test, R_predicted).ToString("0.0000")));

            // TopN Evaluation
            if (topN != 0)
            {
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, topN);
                for (int n = 1; n <= topN; n++)
                {
                    log.AppendLine(Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(RelevantItemsByUser, topNItemsByUser, n).ToString("0.0000")));
                }
            }

            // Save OMFDistribution to file
            if(!File.Exists("RatingOMF_" + GetDataFileName()))
            {
                Utils.IO<Dictionary<Tuple<int, int>, List<double>>>.SaveObject(OMFDistributionByUserItem, "RatingOMF_" + GetDataFileName());
            }

            return log.ToString();
        }
        #endregion

        #region MML
        /*
            Utils.PrintHeading("MML");
            if (Utils.Ask())
            {
                // load the data
                Utils.WriteMovieLens(R_train, "R_train_1m.data");
                Utils.WriteMovieLens(R_test, "R_test_1m.data");
                var training_data = RatingData.Read("R_train_1m.data");
                var test_data = RatingData.Read("R_test_1m.data");

                var m_data = RatingData.Read("1m_comma.data");
                var k_data = RatingData.Read("100k_comma.data");


                var mf = new MatrixFactorization() { Ratings = m_data };
                Console.WriteLine("CV on 1m all data "+mf.DoCrossValidation());
                mf = new MatrixFactorization() { Ratings = k_data };
                Console.WriteLine("CV on 100k all data " + mf.DoCrossValidation());
                mf = new MatrixFactorization() { Ratings = training_data };
                Console.WriteLine("CV on 1m train data " + mf.DoCrossValidation());
                mf = new MatrixFactorization() { Ratings = k_data };
                Console.WriteLine("CV on 100k train data " + mf.DoCrossValidation());


                var bmf = new BiasedMatrixFactorization { Ratings = training_data };
                Console.WriteLine("BMF CV on 1m train data " + bmf.DoCrossValidation());

                // set up the recommender
                var recommender = new MatrixFactorization();// new UserItemBaseline();
                recommender.Ratings = training_data;
                recommender.Train();
                RatingMatrix R_predicted = new RatingMatrix(R_test.UserCount, R_test.ItemCount);
                foreach (var element in R_test.Matrix.EnumerateIndexed(Zeros.AllowSkip))
                {
                    int indexOfUser = element.Item1;
                    int indexOfItem = element.Item2;
                    R_predicted[indexOfUser, indexOfItem] = recommender.Predict(indexOfUser, indexOfItem);
                }

                // Evaluation
                Utils.PrintValue("RMSE of MF on 1m train data, mine RMSE", 
                    RMSE.Evaluate(R_test, R_predicted).ToString("0.0000"));
                var topNItemsByUser = ItemRecommendationCore.GetTopNItemsByUser(R_predicted, Config.TopN);

                Dictionary<int, List<int>> relevantItemsByUser2 = ItemRecommendationCore
    .GetRelevantItemsByUser(R_test, Config.Ratings.RelevanceThreshold);

                for (int n = 1; n <= Config.TopN; n++)
                {
                    Utils.PrintValue("NCDG@" + n, NCDG.Evaluate(relevantItemsByUser2, topNItemsByUser, n).ToString("0.0000"));
                }


                // measure the accuracy on the test data set
                var results = recommender.Evaluate(test_data);
                Console.WriteLine("1m train/test, Their RMSE={0} MAE={1}", results["RMSE"], results["MAE"]);
                Console.WriteLine(results);


            }
         */
        #endregion

    }
}
