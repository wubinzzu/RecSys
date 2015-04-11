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

        #region Compute similarities
        public DenseMatrix UserPearson()
        {
            return ComputeSimilarities(Metric.SimilarityMetric.PearsonRating);
        }
        public DenseMatrix UserCosine()
        {
            throw new NotImplementedException();
        }
        public DenseMatrix ItemPearson()
        {
            throw new NotImplementedException();
        }
        public DenseMatrix ItemCosine()
        {
            throw new NotImplementedException();
        }

        #region Details of similarity calculations
        private DenseMatrix ComputeSimilarities(Metric.SimilarityMetric similarityMetric)
        {
            int dimension = UserCount;

            // For Pearson and Cosine, the max is 1.
            // need to change for other measures
            DenseMatrix similarities = DenseMatrix.OfMatrix(DenseMatrix.Build.DenseDiagonal(dimension, 1));

            // Compute similarity for the lower triangular
            Object lockMe = new Object();
            Parallel.For(0, dimension, i =>
            {
                Utils.PrintEpoch("Progress user/total", i, dimension);

                for (int j = 0; j < dimension; j++)
                {
                    if (i == j) { continue; }// Skip the diagonal
                    else if (i > j)
                    {
                        switch (similarityMetric)
                        {
                            case Metric.SimilarityMetric.CosineRating:
                                double cosine = Metric.CosineR(this, i, j);
                                lock (lockMe)
                                {
                                    similarities[i, j] = cosine;
                                }
                                break;
                            case Metric.SimilarityMetric.PearsonRating:
                                double pearson = Metric.PearsonR(this, i, j);
                                lock (lockMe)
                                {
                                    similarities[i, j] = pearson;
                                }
                                break;
                        }
                    }
                }
            });
            // Copy similarity values from lower triangular to upper triangular
            similarities = DenseMatrix.OfMatrix(similarities + similarities.Transpose()
                - DenseMatrix.CreateIdentity(similarities.RowCount));

            Debug.Assert(similarities[0, 0] == 1, "The similarities[0,0] should be 1 for Pearson correlation.");

            return similarities;
        }
        #endregion
        #endregion

        #region DatasetBrief
        public string DatasetBrief(string title)
        {
            string brief = "";
            brief += Utils.CreateHeading(title);
            brief += string.Format("{0,-23} │ {1,13}\n", "# of users", UserCount);
            brief += string.Format("{0,-23} │ {1,13}\n", "# of items", ItemCount);
            brief += string.Format("{0,-23} │ {1,13}\n", "# of ratings", NonZerosCount);
            brief += string.Format("{0,-23} │ {1,13:0.00}\n", "Density level", Density);
            brief += string.Format("{0,-23} │ {1,13:0.00}\n", "Global mean", GetGlobalMean());
            brief += Config.BottomRule;
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
