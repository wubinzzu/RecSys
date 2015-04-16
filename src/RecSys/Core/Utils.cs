using MathNet.Numerics.Data.Text;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Numerical;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace RecSys
{
    /// <summary>
    /// This class implements core functions shared by differnet algorithms.
    /// Including read/write files, printing messages, timer, etc.
    /// </summary>
    public class Utils
    {
        #region Data IO

        #region Load movielens dataset into SparseMatrix
        /// <summary>
        /// Load movielens data set, the data set will be split into train and test sets.
        /// Pre-shuffle the file and swith off shuffle option is recommended for large data set.
        /// </summary>
        /// <param name="fileOfDataSet">Path to the movielens data set.</param>
        /// <param name="R_train">The training set will be sent out from this parameter.</param>
        /// <param name="R_test">The testing set will be sent out from this parameter.</param>
        /// <param name="minCountOfRatings">Users with ratings less than the specified count 
        /// will be excluded from the data set.</param>
        /// <param name="countOfRatingsForTrain">Specifies how many ratings for each user to 
        /// keep in the training set, and the reset in the testing set.</param>
        /// <param name="shuffle">Specifies whether the lines in the file should be read 
        /// in random order or not.</param>
        /// <param name="seed">The random seed for shuffle.</param>
        public static void LoadMovieLensSplitByCount(string fileOfDataSet, out RatingMatrix R_train,
            out RatingMatrix R_test, int minCountOfRatings = Config.MinCountOfRatings,
            int countOfRatingsForTrain = Config.CountOfRatingsForTrain, bool shuffle = false, int seed = 1)
        {
            Dictionary<int, int> userByIndex = new Dictionary<int, int>();   // Mapping from index in movielens file to user index in matrix
            Dictionary<int, int> ratingCountByUser = new Dictionary<int, int>(); // count how many ratings of each user
            Dictionary<int, int> itemByIndex = new Dictionary<int, int>();   // Mapping from index in movielens file to item index in matrix

            // Read the file to discover the whole matrix structure and mapping
            foreach (string line in File.ReadLines(fileOfDataSet))
            {
                string[] tokens = line.Split(Config.SplitSeperators, StringSplitOptions.RemoveEmptyEntries);
                int indexOfUser = int.Parse(tokens[0]);
                int indexOfItem = int.Parse(tokens[1]);
                if (!userByIndex.ContainsKey(indexOfUser))          // We update index only for new user
                {
                    userByIndex[indexOfUser] = userByIndex.Count;   // The current size is just the current matrix index
                    ratingCountByUser[indexOfUser] = 1;             // Initialize the rating count for this new user
                }
                else { ratingCountByUser[indexOfUser]++; }

                if (!itemByIndex.ContainsKey(indexOfItem))          // We update index only for new item
                {
                    itemByIndex[indexOfItem] = itemByIndex.Count;   // The current size is just the current matrix index
                }
            }

            // Remove users with too few ratings
            int countOfRemovedUsers = 0;
            List<int> indexes = userByIndex.Keys.ToList();
            foreach (int fileIndexOfUser in indexes)
            {
                if (ratingCountByUser[fileIndexOfUser] < minCountOfRatings)
                {
                    int indexOfRemovedUser = userByIndex[fileIndexOfUser];
                    userByIndex.Remove(fileIndexOfUser);
                    List<int> keys = userByIndex.Keys.ToList();
                    // We need to shift the matrix index by 1 after removed one user
                    foreach (int key in keys)
                    {
                        if (userByIndex[key] > indexOfRemovedUser)
                        {
                            userByIndex[key] -= 1;
                        }
                    }
                    countOfRemovedUsers++;
                }
            }

            Console.WriteLine(countOfRemovedUsers + " users have less than " + minCountOfRatings + " and were removed.");

            R_train = new RatingMatrix(userByIndex.Count, itemByIndex.Count);
            R_test = new RatingMatrix(userByIndex.Count, itemByIndex.Count);

            // Read file data into rating matrix
            Dictionary<int, int> trainCountByUser = new Dictionary<int, int>(); // count how many ratings in the train set of each user

            // Create a enumerator to enumerate each line in the file
            IEnumerable<string> linesInFile;
            if (shuffle)
            {
                Random rng = new Random(seed);
                var allLines = new List<string>(File.ReadAllLines(fileOfDataSet));
                allLines.Shuffle(rng);
                linesInFile = allLines.AsEnumerable<string>();
            }
            else
            {
                linesInFile = File.ReadLines(fileOfDataSet);
            }

            // Process each line and put ratings into training/testing sets
            foreach (string line in linesInFile)
            {
                string[] tokens = line.Split(Config.SplitSeperators, StringSplitOptions.RemoveEmptyEntries);
                int fileIndexOfUser = int.Parse(tokens[0]);
                int fileIndexOfItem = int.Parse(tokens[1]);
                double rating = double.Parse(tokens[2]);
                if (userByIndex.ContainsKey(fileIndexOfUser))   // If this user was not removed
                {
                    int indexOfUser = userByIndex[fileIndexOfUser];
                    int indexOfItem = itemByIndex[fileIndexOfItem];
                    if (!trainCountByUser.ContainsKey(indexOfUser))
                    {
                        // Fill up the train set
                        R_train[indexOfUser, indexOfItem] = rating;
                        trainCountByUser[indexOfUser] = 1;
                    }
                    else if (trainCountByUser[indexOfUser] < countOfRatingsForTrain)
                    {
                        // Fill up the train set
                        R_train[indexOfUser, indexOfItem] = rating;
                        trainCountByUser[indexOfUser]++;
                    }
                    else
                    {
                        // Fill up the test set
                        R_test[indexOfUser, indexOfItem] = rating;
                    }
                }
            }

            Debug.Assert(userByIndex.Count * countOfRatingsForTrain == R_train.NonZerosCount);
        }
        #endregion

        /// <summary>
        /// Write a matrix (sparse or dense) to a comma separated file.
        /// </summary>
        /// <param name="matrix">The matrix to be written.</param>
        /// <param name="path">Path of output file.</param>
        public static void WriteMatrix(Matrix<double> matrix, string path)
        {
            DelimitedWriter.Write(path, matrix, ",");
        }

        /// <summary>
        /// Read a desen matrix from file. 0 values are stored.
        /// </summary>
        /// <param name="path">Path of input file.</param>
        /// <returns>Matrix filled with data from file.</returns>
        public static Matrix<double> ReadDenseMatrix(string path)
        {
            return DelimitedReader.Read<double>(path, false, ",", false);
        }

        /// <summary>
        /// Read a sparse matrix from file. 0 values are ignored.
        /// </summary>
        /// <param name="path">Path of input file.</param>
        /// <returns>A SparseMatrix.</returns>
        public static SparseMatrix ReadSparseMatrix(string path)
        {
            return SparseMatrix.OfMatrix(DelimitedReader.Read<double>(path, false, ",", false));
        }

        /// <summary>
        /// Create a Matrix filled with random numbers from [0,1], uniformly distributed.
        /// </summary>
        /// <param name="rowCount">Number of rows.</param>
        /// <param name="columnCount">Number of columns.</param>
        /// <param name="seed">Random seed.</param>
        /// <returns>A Matrix filled with random numbers from [0,1].</returns>
        public static Matrix<double> CreateRandomMatrix(int rowCount, int columnCount, int seed = Config.Seed)
        {
            ContinuousUniform uniformDistribution = new ContinuousUniform(0, 1, new Random(Config.Seed));
            Matrix<double> randomMatrix = Matrix.Build.Random(rowCount, columnCount, uniformDistribution);

            Debug.Assert(randomMatrix.Find(x => x > 1 && x < 0) == null);  // Check the numbers are in [0,1]

            return randomMatrix;
        }
        #endregion

        #region String formatting and printing
        public static string CreateHeading(string title)
        {
            string formatedTitle = "";
            formatedTitle += new String('*', Config.RightPad + Config.LeftPad + 2) + "\n";
            formatedTitle += title.PadLeft((Config.RightPad + Config.LeftPad + title.Length) / 2, ' ') + "\n";
            formatedTitle += new String('*', Config.RightPad + Config.LeftPad + 2) + "\n";
            return formatedTitle;
        }

        public static void PrintValue(string label, string value)
        {
            string formatedString = "";
            string labelToPrint = label;
            while (labelToPrint.Length > Config.RightPad)
            {
                formatedString += labelToPrint.Substring(0,Config.RightPad) + "│\n";
                labelToPrint = labelToPrint.Remove(0, Config.RightPad);
            }
            formatedString += String.Format("{0}│{1}", 
                labelToPrint.PadRight(Config.RightPad, ' '),
                value.PadLeft(Config.LeftPad, ' '));
            Console.WriteLine(formatedString);
        }

        public static string PrintValueToString(string label, string value)
        {
            return string.Format("{0}│{1}", label.PadRight(Config.RightPad, ' '),
                value.PadLeft(Config.LeftPad, ' '));
        }

        public static void PrintHeading(string title)
        {
            Console.Write(CreateHeading(title));
        }

        public static void PrintEpoch(string label, int epoch, int maxEpoch)
        {
            if (epoch == 0 || epoch == maxEpoch - 1 || epoch % (int)Math.Ceiling(maxEpoch * 0.1) == 4)
            {
                PrintValue(label, (epoch + 1) + "/" + maxEpoch);
            }
        }
        public static void PrintEpoch(string label1, int epoch, int maxEpoch, string label2, double error)
        {
            if (epoch == 0 || epoch == maxEpoch - 1 || epoch % (int)Math.Ceiling(maxEpoch * 0.1) == 4)
            {
                PrintValue(label2 + "@" + label1 + " (" + (epoch + 1) + "/" + maxEpoch + ")", error.ToString("0.0000"));
            }
        }
        public static void PrintEpoch(string label1, int epoch, int maxEpoch, string label2, string message)
        {
            if (epoch == 0 || epoch == maxEpoch - 1 || epoch % (int)Math.Ceiling(maxEpoch * 0.1) == 4)
            {
                PrintValue(label2 + "@" + label1 + " (" + (epoch + 1) + "/" + maxEpoch + ")", message);
            }
        }
        #endregion

        #region Timer & Excution control
        private static Stopwatch stopwatch;
        public static void StartTimer()
        {
            stopwatch = Stopwatch.StartNew();
        }

        public static void StopTimer()
        {
            stopwatch.Stop();
            double seconds = stopwatch.Elapsed.TotalMilliseconds / 1000;
            Console.WriteLine("{0}│{1}s", "Computation time".PadRight(Config.RightPad, ' '),
                seconds.ToString("0.000").PadLeft(Config.LeftPad - 1, ' '));
        }

        public static void Pause()
        {
            Console.WriteLine("\nPress any key to continue...");
            Console.ReadKey();
            Console.SetCursorPosition(0, Console.CursorTop - 2);
            Console.Write(new String(' ', Console.BufferWidth));
        }

        public static bool Ask()
        {
            Console.WriteLine("\nPress 'S' to skip or any key to run...");
            ConsoleKeyInfo key = Console.ReadKey();
            if (key.Key == ConsoleKey.S)
            {
                Console.WriteLine("Skipped.");
                return false;
            }
            else
            {
                Console.SetCursorPosition(0, Console.CursorTop - 2);
                Console.Write(new String(' ', Console.BufferWidth));
                return true;
            }
        }
        #endregion


        #region Load OMF
        public static Dictionary<Tuple<int, int>, double[]> LoadOMFDistributions(string fileName)
        {
            Dictionary<Tuple<int, int>, double[]> OMFDistributions = new Dictionary<Tuple<int, int>, double[]>();

            // Read the file to discover the whole matrix structure and mapping
            foreach (string line in File.ReadLines(fileName))
            {
                string[] tokens = line.Split(Config.SplitSeperators, StringSplitOptions.RemoveEmptyEntries);
                int indexOfUser = int.Parse(tokens[0]);
                int indexOfItem = int.Parse(tokens[1]);
                // TODO: make it dynamic later
                OMFDistributions[new Tuple<int, int>(indexOfUser, indexOfItem)] = new double[] {
                    double.Parse(tokens[2]),
                    double.Parse(tokens[3]),
                    double.Parse(tokens[4])};
            }

            return OMFDistributions;
        }
        #endregion

        #region Obsolete
        [Obsolete("LoadMovieLens(string path) is deprecated.")]
        public static RatingMatrix LoadMovieLens(string path)
        {
            RatingMatrix R;

            Dictionary<int, int> userMap = new Dictionary<int, int>();   // Mapping from index in movielens file to index in matrix
            Dictionary<int, int> itemMap = new Dictionary<int, int>();   // Mapping from index in movielens file to index in matrix

            foreach (string line in File.ReadLines(path))
            {
                string[] tokens = line.Split('\t');
                int t1 = int.Parse(tokens[0]);
                int t2 = int.Parse(tokens[1]);
                if (!userMap.ContainsKey(t1))    // We update index only for new user
                {
                    userMap[t1] = userMap.Count;// The current size is just the current matrix index
                }
                if (!itemMap.ContainsKey(t2))// We update index only for new item
                {
                    itemMap[t2] = itemMap.Count;// The current size is just the current matrix index
                }
            }

            R = new RatingMatrix(userMap.Count, itemMap.Count);

            foreach (string line in File.ReadLines(path))
            {
                string[] tokens = line.Split('\t');
                int uid = userMap[int.Parse(tokens[0])];
                int iid = itemMap[int.Parse(tokens[1])];
                double rating = double.Parse(tokens[2]);
                R[uid, iid] = rating;
            }
            return R;
        }
        #endregion
    }

    #region My extensions to .Net and Math.Net libraries
    public static class ExtensionsToDotNet
    {
        /// <summary>
        /// Add a function to IList interfance to shuffle the list with Fisher–Yates shuffle.
        /// See http://stackoverflow.com/questions/273313/randomize-a-listt-in-c-sharp
        /// and http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        public static void Shuffle<T>(this IList<T> list, Random rng)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }

    public static class ExtensionsToMathNet
    {
        /// <summary>
        /// An extension function to compute the sum of squares of non-zero elements
        /// </summary>
        public static double SquaredSum(this Matrix<double> matrix)
        {
            return matrix.PointwisePower(2).RowSums().Sum();
            //return Math.Pow(matrix.FrobeniusNorm(), 2);
        }

        public static int GetNonZerosCount(this Vector<double> vector)
        {
            //Debug.Assert(!vector.Storage.IsDense);
            return ((SparseVector)vector).NonZerosCount;
        }

        public static int GetNonZerosCount(this Matrix<double> matrix)
        {
            //Debug.Assert(!vector.Storage.IsDense);
            return ((SparseMatrix)matrix).NonZerosCount;
        }
    }
    #endregion
}
