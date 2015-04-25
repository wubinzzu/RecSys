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
using MathNet.Numerics.Statistics;
using System.IO;
using MyMediaLite.IO;
using MyMediaLite.RatingPrediction;
using MyMediaLite.Eval;
using RecSys.Experiments;

namespace RecSys
{
    class Program
    {
        static void Main(string[] args)
        {
            //ExperimentOfSpeed.SpeedOfAccessRandomElement();
            Control.UseMultiThreading();
            //CIKMExpDesign.NMFonMovieLens20M();
            ExperimentEngine aExperiment = new ExperimentEngine("MovieLens100K.data", 60, 500, 50, true, 1, 5.0, 200, 0.2);
            aExperiment.GetReadyForOrdinal();
            aExperiment.GetReadyForNumerical();
            aExperiment.GetReadyAll();
            //aExperiment.RunPrefNMF(50, 0.001, 0.001, 0.0005, 50, 10);
            //aExperiment.RunPrefNMFbasedOMF(50, 0.001, 0.001, 0.0005, 50, Config.Preferences.quantizerThree, 10);
            aExperiment.RunPrefMRF(0.1, 0.01, 300, Config.Preferences.quantizerThree, 10);

            aExperiment.RunNMF(100, 0.1, 0.15, 50, 10);
            aExperiment.RunNMFbasedOMF(100, 0.1, 0.15, 50, Config.Preferences.quantizerFive, 10);
            aExperiment.RunNMFbasedORF(0.05, 0.01, 100, Config.Preferences.quantizerFive, 10);

            aExperiment.RunGlobalMean();
            aExperiment.RunMostPopular(10);
            aExperiment.RunUserKNN(10);
            aExperiment.RunPrefKNN(10);

            Utils.Pause();

            //Utils.RemoveColdUsers(70, "MovieLens20M.data");

            Utils.Pause();
        }
    }
}
