using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Experiments
{
    public class CIKMExpDesign
    {
        // Common configurations
        static int neighborCount = 50;
        static int factorCount = 50;
        static int defaultMaxEpoch = 100;
        static string MovieLens20MFile = "MovieLens1M.data";//"100k.data";
        static List<int> GivenSizes = new List<int>() { 50, 60 };
        static int minTestSize = 10;
        static bool shuffle = true;
        static double relevantCriteria = 5;
        static double defaultLearnRate = 0.1;
        static double defaultRegularization = 0.05;
        static int topN = 10;

        /**********************************************************
         * Experiment 1
         * 
         * This experiment is to evaluate the top-N recommendation 
         * performance of NMF on MovieLens 20M data set with default 
         * factor count = 50
         * However, the Given data size is varied from 30, 40, 50, 60
         * Repeat ten times
         * ********************************************************/
        public static void NMFonMovieLens20M()
        {
            for(int seed = 1; seed <= 10;  seed++)
            {
                foreach(int givenSize in GivenSizes)
                {

                    ExperimentEngine experiment = new ExperimentEngine(
                        MovieLens20MFile,
                        givenSize + minTestSize,
                        givenSize,
                        shuffle,
                        seed,
                        relevantCriteria,
                        neighborCount,
                        seed * 0.1);

                    string log = experiment.RunNMF(defaultMaxEpoch, defaultLearnRate, defaultRegularization, factorCount, topN);
                    using (StreamWriter w = File.AppendText("NMFonMovieLens20M_Log.txt"))
                    {
                        w.WriteLine("=========================================" + seed + "/" + givenSize);
                        w.WriteLine(log);
                    }
                }
            }
        }
    }
}
