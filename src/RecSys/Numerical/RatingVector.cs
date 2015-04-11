using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Numerical
{
    class RatingVector
    {
        SparseVector ratingVector;
        public RatingVector(Vector<double> ratingVector)
        {
            this.ratingVector = SparseVector.OfVector(ratingVector);
        }

        public IEnumerable<Tuple<int, double>> Ratings
        {
            get { return ratingVector.EnumerateIndexed(Zeros.AllowSkip); }
        }
    }
}
