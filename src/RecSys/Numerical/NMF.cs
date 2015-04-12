using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Numerical;
using System;

namespace RecSys
{
    /// <summary>
    /// The Non-negative Matrix Factorization
    /// See Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. NIPS.
    /// and Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer.
    /// </summary>
    public class NMF
    {
        public static RatingMatrix PredictRatings(RatingMatrix R_train, RatingMatrix R_unknown,
            int maxEpoch, double learnRate, double regularization, int factorCount)
        {
            int userCount = R_train.UserCount;
            int itemCount = R_train.ItemCount;
            int ratingCount = R_train.NonZerosCount;

            // User latent vectors with default seed
            DenseMatrix P = Utils.CreateRandomDenseMatrix(userCount, factorCount, Config.Seed);
            // Item latent vectors with a different seed
            DenseMatrix Q = Utils.CreateRandomDenseMatrix(factorCount, itemCount, Config.Seed + 1);

            // SGD
            for (int epoch = 0; epoch < maxEpoch; ++epoch)
            {
                foreach (Tuple<int, int, double> element in R_train.Ratings)
                {
                    int userIndex = element.Item1;
                    int itemIndex = element.Item2;
                    double rating = element.Item3;

                    double e_ij = rating - P.Row(userIndex).DotProduct(Q.Column(itemIndex));

                    // Update feature vectors
                    for (int k = 0; k < factorCount; ++k)
                    {
                        double factorOfUser = P[userIndex, k];
                        double factorOfItem = Q[k, itemIndex];

                        P[userIndex, k] += learnRate * (e_ij * factorOfItem - regularization * factorOfUser);
                        Q[k, itemIndex] += learnRate * (e_ij * factorOfUser - regularization * factorOfItem);
                    }
                }

                // Display the current regularized error see if it converges
                double e_prev = double.MaxValue;
                double e_curr = 0;
                if (epoch == 0 || epoch == maxEpoch - 1 || epoch % (int)Math.Ceiling(maxEpoch * 0.1) == 4)
                {
                    double e = 0;
                    foreach (Tuple<int, int, double> element in R_train.Ratings)
                    {
                        int userIndex = element.Item1;
                        int itemIndex = element.Item2;
                        double rating = element.Item3;

                        e += Math.Pow(rating - P.Row(userIndex).DotProduct(Q.Column(itemIndex)), 2);
                        for (int k = 0; k < factorCount; ++k)
                        {
                            e += (regularization / 2) * (Math.Pow(P[userIndex, k], 2) + Math.Pow(Q[k, itemIndex], 2));
                        }
                    }

                    // Record the current error
                    e_curr = e;

                    Utils.PrintEpoch("Epoch", epoch, maxEpoch, "Learning error", Math.Sqrt(e / ratingCount));
                }
                // Stop the learning if the regularized error falls below a certain threshold
                if (e_prev - e_curr < 0.001)
                {
                    Console.WriteLine("Improvment less than 0.001, learning stopped.");
                    break;
                }
                e_prev = e_curr;
            }
            return new RatingMatrix(R_unknown.Matrix.PointwiseMultiply(P.Multiply(Q)));
        }
    }
}
