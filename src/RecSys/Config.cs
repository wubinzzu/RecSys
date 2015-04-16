using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys
{
    public class Config
    {
        //========Environment settings=======
        public const int Seed = 2;
        public static readonly int LeftPad = 13;
        public static readonly int RightPad = 35;
        public static readonly string Rule = "\n───────────────────────\n"; // \u2500
        public static readonly string LongRule = "\n──────────────────────────────────\n";
        public static readonly bool RunNMF = true;
        public static readonly bool RunPrefNMF = true;
        public static readonly bool RunRatingUserKNN = true;
        public static readonly bool RunPreferenceUserKNN = true;
        public static readonly bool RunGlobalMean = true;
        public static readonly bool LoadSavedData = true;
        public static readonly double ZeroInSparseMatrix = -99;//1e-14;
        public const int MinCountOfRatings = 60;
        public const int CountOfRatingsForTrain = 50;
        public static readonly string[] SplitSeperators = { "\t", "::" };

        public static readonly string UserSimilaritiesOfRatingFile = "userSimilaritiesOfRating.csv";
        public static readonly string UserSimilaritiesOfPrefFile = "userSimilaritiesOfPref.csv";
        public static readonly string ItemSimilaritiesOfRatingFile = "itemSimilaritiesOfRating.csv";
        public static readonly string ItemSimilaritiesOfPrefFile = "itemSimilaritiesOfPref.csv";

        public class OMF
        {
            public static readonly int MaxEpoch = 1000;
            public static readonly double LearnRate = 0.001;
            public static readonly double Regularization = 0.015;
            public static readonly int LevelCount = 5;
        }

        public class NMF
        {
            // Matrix Factorization
            public static readonly int K = 30;				// Num of factors
            public static readonly int MaxEpoch = 100;
            public static readonly double LearnRate = 0.01;
            public static readonly double Regularization = 0.1;
        }

        public class PrefNMF
        {
            // Matrix Factorization
            public static readonly int K = 30;				// Num of factors
            public static readonly int MaxEpoch = 30;
            public static readonly double LearnRate = 0.001;
            public static readonly double RegularizationOfUser = 0.05;
            public static readonly double RegularizationOfItem = 0.03;
        }

        public class Ratings
        {
            public static readonly string DataSetFile = "100k.data";
            public static readonly string TrainSetFile = "R_train.csv";
            public static readonly string TestSetFile = "R_test.csv";
            public static readonly double RelevanceThreshold = 5.0;   // Only 5-star items are considered as relevant
            public static readonly double MaxRating = 5.0;
            public static readonly double MinRating = 1.0;
        }

        public class Preferences
        {
            // The position should be in [-1,1] but due to the difficult of storing 0 in sparse matrix
            // we shift all position values by 2 so the range becomes [1-3]
            public static readonly double PositionShift = 2;
            public static readonly double Preferred = 3;
            public static readonly double EquallyPreferred = 2;
            public static readonly double LessPreferred = 1;
            public static readonly List<double> quantizerFive = new List<double> { 1, 2, 3, 4, 5 };
            public static readonly List<double> quantizerThree = new List<double> { 1, 2, 3};
        }

        public class KNN
        {
            public static readonly int K = 10;		// Num of neighbors
        }
        public static readonly int TopN = 10;
    }

}
