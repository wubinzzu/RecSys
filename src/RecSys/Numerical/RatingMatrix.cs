using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Core;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Numerical
{
    public class RatingMatrix
    {
        #region Variables
        private SparseMatrix ratingMatrix;
        private DenseMatrix userSimilarities;
        #endregion

        #region Properties
        public int UserCount { get { return ratingMatrix.RowCount; } }
        public int ItemCount { get { return ratingMatrix.ColumnCount; } }
        public int NonZerosCount { get { return ratingMatrix.NonZerosCount; } }
        public double Density { get { return (double)NonZerosCount / (UserCount * ItemCount); } }
        public SparseMatrix Matrix { get { return ratingMatrix; } }
        public DenseMatrix UserSimilarities
        {
            get
            {
                Debug.Assert(!(userSimilarities == null));
                return userSimilarities;
            }
            set
            {
                userSimilarities = value;
            }
        }
        public double this[int indexOfUser, int indexOfItem]
        {
            get { return ratingMatrix[indexOfUser, indexOfItem]; }
            set { ratingMatrix[indexOfUser, indexOfItem] = value; }
        }

        public SparseVector this[int userIndex]
        {
            get { return SparseVector.OfVector(ratingMatrix.Row(userIndex)); }
        }

        public IEnumerable<Tuple<int, Vector<double>>> Users
        {
            get { return ratingMatrix.EnumerateRowsIndexed(); }
        }
        public IEnumerable<Tuple<int, int, double>> Ratings
        {
            get { return ratingMatrix.EnumerateIndexed(Zeros.AllowSkip); }
        }
        #endregion

        #region Constructors
        public RatingMatrix(int userCount, int itemCount)
        {
            ratingMatrix = new SparseMatrix(userCount, itemCount);
        }
        public RatingMatrix(SparseMatrix ratingMatrix)
        {
            this.ratingMatrix = ratingMatrix;
        }
        public RatingMatrix(Matrix<double> ratingMatrix)
        {
            this.ratingMatrix = SparseMatrix.OfMatrix(ratingMatrix);
        }
        #endregion

        #region DatasetBrief
        public string DatasetBrief(string title)
        {
            string brief = "";
            brief += Utils.CreateHeading(title);
            brief += Utils.PrintValueToString("# of users", UserCount.ToString("D")) + "\n";
            brief += Utils.PrintValueToString("# of items", ItemCount.ToString("D")) + "\n";
            brief += Utils.PrintValueToString("# of ratings", NonZerosCount.ToString("D")) + "\n";
            brief += Utils.PrintValueToString("Density level", Density.ToString("P")) + "\n";
            brief += Utils.PrintValueToString("Global mean", GetGlobalMean().ToString("0.00"));
            return brief;
        }
        #endregion

        #region Other methods
        // Returns the average of all known ratings
        public double GetGlobalMean()
        {
            return ratingMatrix.RowSums().Sum() / ratingMatrix.NonZerosCount;
        }
        // Returns the average ratings of each user
        public Vector<double> GetUserMeans()
        {
            return ratingMatrix.RowSums() / (ratingMatrix.PointwiseDivide(ratingMatrix).RowSums());
        }

        // Returns the average ratings of each item
        public Vector<double> GetItemMeans()
        {
            return ratingMatrix.ColumnSums() / (ratingMatrix.PointwiseDivide(ratingMatrix).ColumnSums());
        }

        public SparseVector GetRow(int userIndex)
        {
            return SparseVector.OfVector(ratingMatrix.Row(userIndex));
        }

        public RatingMatrix IndexesOfNonZeroElements()
        {
            return new RatingMatrix(ratingMatrix.PointwiseDivide(ratingMatrix));
        }

        public RatingMatrix Multiply(double scalar)
        {
            return new RatingMatrix(ratingMatrix.Multiply(scalar));
        }
        #endregion
    }
}
