using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Experiments
{
    class ExperimentOfSpeed
    {

        #region Speed of accessing rows
        /// <summary>
        /// List indexing is 2000x faster than Matrix.Row() or enumeration.
        /// </summary>
        public static void SpeedOfGetRow()
        {
            SparseMatrix myMatrix = new SparseMatrix(1000, 1000);
            SparseVector myVector = SparseVector.OfVector(Vector.Build.Random(1000));
            myVector.CoerceZero(1.8);
            for (int i = 0; i < 1000; i++)
            {
                myMatrix.SetRow(i, myVector);
            }
            List<Vector<double>> myList = new List<Vector<double>>(myMatrix.EnumerateRows());

            Utils.StartTimer();
            for (int repeat = 0; repeat < 10; repeat++)
            {
                for (int i = 0; i < 1000; i++)
                {
                    double foo = myMatrix.Row(i)[0];
                }
            }
            Utils.StopTimer();

            Utils.StartTimer();
            for (int repeat = 0; repeat < 10; repeat++)
            {
                foreach(var row in myMatrix.EnumerateRowsIndexed())
                {
                    double foo = row.Item2[0];
                }
            }
            Utils.StopTimer();

            Utils.StartTimer();
            for (int repeat = 0; repeat < 10; repeat++)
            {
                for (int i = 0; i < 1000; i++)
                {
                    double foo = myList[i][0];
                }
            }
            Utils.StopTimer();
        }
        #endregion

        #region Speed of random access
        /// <summary>
        /// 2D nested list is 10x faster than SparseMatrix and List<SparseVector>
        /// and 100x faster than Dictionary. However, 2D nested list is only for dense case.
        /// Note that Nested Dictionary is faster than Tuple dictionary! 
        /// it is the second fastest only 4 times slower than  2D list.
        /// </summary>
        public static void SpeedOfAccessRandomElement()
        {
            SparseMatrix myMatrix = new SparseMatrix(1000, 1000);
            SparseVector myVector = SparseVector.OfVector(Vector.Build.Random(1000));
            myVector.CoerceZero(1.8);
            for (int i = 0; i < 1000; i++)
            {
                myMatrix.SetRow(i, myVector);
            }

            List<Vector<double>> myList = new List<Vector<double>>(myMatrix.EnumerateRows());
            List<List<double>> my2DList = new List<List<double>>();

            Dictionary<Tuple<int, int>, double> myDict = new Dictionary<Tuple<int, int>, double>();
            Dictionary<int, Dictionary<int, double>> myDict2 = 
                new Dictionary<int, Dictionary<int, double>>();
            for (int i = 0; i < 1000; i++)
            {
                myDict2[i] = new Dictionary<int, double>();
                for (int j = 0; j < 1000; j++)
                {
                    myDict[new Tuple<int, int>(i, j)] = i;
                    myDict2[i][j] = i;
                }
            }

            for (int i = 0; i < 1000; i++)
            {
                my2DList.Add(new List<double>());
                for (int j = 0; j < 1000; j++)
                {
                    my2DList[i].Add(i);
                }
            }

            Utils.StartTimer();
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                {
                    double foo = myDict[new Tuple<int, int>(i, j)];
                }
            }
            Utils.StopTimer();

            Utils.StartTimer();
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                {
                    double foo = myDict2[i][j];
                }
            }
            Utils.StopTimer();

            Utils.StartTimer();
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                {
                    double foo = myMatrix[i, j];
                }
            }
            Utils.StopTimer();

            Utils.StartTimer();
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                {
                    double foo = myList[i][j];
                }
            }
            Utils.StopTimer();

            Utils.StartTimer();
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                {
                    double foo = my2DList[i][j];
                }
            }
            Utils.StopTimer();

            Utils.Pause();
        }
        #endregion
    }
}
