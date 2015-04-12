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
        public static readonly bool LoadSavedData = false;
        public static readonly double ZeroInSparseMatrix = 1e-14;
        public const int MinCountOfRatings = 60;
        public const int CountOfRatingsForTrain = 50;

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
            public static readonly int K = 10;				// Num of factors
            public static readonly int MaxEpoch = 100;
            public static readonly double LearnRate = 0.01;
            public static readonly double RegularizationOfUser = 0.001;     // From Desarkar's paper
            public static readonly double RegularizationOfItem = 0.0005;    // From Desarkar's paper
        }

        public class Ratings
        {
            public static readonly string DataSetFile = "u.data";
            public static readonly string TrainSetFile = "R_train.csv";
            public static readonly string TestSetFile = "R_test.csv";
            public static readonly string UserSimilaritiesOfRatingFile = "userSimilaritiesOfRating.csv";
            public static readonly string UserSimilaritiesOfPrefFile = "userSimilaritiesOfPref.csv";
            public static readonly double RelevanceThreshold = 5.0;   // Only 5-star items are considered as relevant
            public static readonly double MaxRating = 5.0;
            public static readonly double MinRating = 1.0;
        }

        public class Preferences
        {
            public static readonly double Preferred = 1;
            public static readonly double EquallyPreferred = 0.5;
            public static readonly double LessPreferred = ZeroInSparseMatrix;
        }

        public class KNN
        {
            public static readonly int K = 10;		// Num of neighbors
        }
        public static readonly int TopN = 10;
    }

}
