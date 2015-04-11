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
        public const int Indent = 40;
        public const string Rule = "\n───────────────────────\n"; // \u2500
        public const string LongRule = "\n──────────────────────────────────\n";
        public const string BottomRule = "\n";
        public const bool RunNMF = true;
        public const bool RunRatingUserKNN = true;
        public const bool RunPreferenceUserKNN = true;
        public const bool RunGlobalMean = true;
        public const bool RecomputeSimilarity = true;
        public const double ZeroInSparseMatrix = 1e-14;
        public const int MinCountOfRatings = 60;
        public const int CountOfRatingsForTrain = 50;

        public class NMF
        {
            // Matrix Factorization
            public const int K = 30;				// Num of factors
            public const int MaxEpoch = 100;
            public const double LearnRate = 0.01;
            public const double Regularization = 0.1;
        }

        public class Ratings
        {
            public const string TrainSetFile = "ua.base";
            public const string TestSetFile = "ua.test";
            public const string UserSimilarityFile = "uaSimilaritiesOfUser.data";
            public const double RelevanceThreshold = 5.0;   // Only 5-star items are considered as relevant
            public const double MaxRating = 5.0;
            public const double MinRating = 1.0;
        }

        public class Preferences
        {
            public const double Preferred = 3;
            public const double EquallyPreferred = 2;
            public const double LessPreferred = 1;
        }

        public class KNN
        {
            public const int K = 10;		// Num of neighbors
        }
        public const int TopN = 10;






    }

}
