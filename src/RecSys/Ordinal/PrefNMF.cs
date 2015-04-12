using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Core;
using RecSys.Numerical;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Ordinal
{
    /// <summary>
    /// Implementation of the PrefNMF algorithm described in:
    /// Desarkar, M. S., Saxena, R., & Sarkar, S. (2012). 
    /// Preference relation based matrix factorization for recommender systems. 
    /// In User Modeling, Adaptation, and Personalization (pp. 63-75). Springer Berlin Heidelberg.
    /// </summary>
    public class PrefNMF
    {
        public static RatingMatrix PredictRatings(PrefRelations PR_train, RatingMatrix R_unknown,
            int maxEpoch, double learnRate, double regularization, int factorCount)
        {
            int userCount = PR_train.UserCount;
            int itemCount = PR_train.ItemCount;
            int prefCount = PR_train.GetTotalPrefRelationsCount();

            // User latent vectors with default seed
            DenseMatrix P = Utils.CreateRandomDenseMatrix(userCount, factorCount, Config.Seed);
            // Item latent vectors with a different seed
            DenseMatrix Q = Utils.CreateRandomDenseMatrix(factorCount, itemCount, Config.Seed + 1);

            // SGD
            for (int epoch = 0; epoch < maxEpoch; ++epoch)
            {
                // For each epoch, we will iterate through all 
                // preference relations of all users

                // Loop through each user
                foreach (var pair in PR_train.PreferenceRelationsByUser)
                {
                    int indexOfUser = pair.Key;
                    SparseMatrix preferenceRelationsOfUser = pair.Value;

                    // For each preference relation of this user, update the latent feature vectors
                    foreach (var entry in preferenceRelationsOfUser.EnumerateIndexed(Zeros.AllowSkip))
                    {
                        int indexOfItem_i = entry.Item1;
                        int indexOfItem_j = entry.Item2;
                        double prefRelation_uij = entry.Item3;

                        // TODO: Maybe it can be faster to do two dot products to remove the substraction (lose sparse  property I think)
                        double estimate_uij = P.Row(indexOfUser).DotProduct(Q.Column(indexOfItem_i) - Q.Column(indexOfItem_j));   // Eq. 2
                        double exp_estimate_uij = Math.Exp(estimate_uij);   // enumerator in Eq. 2
                        double normalized_estimate_uij = SpecialFunctions.InverseLogit(estimate_uij);   // pi_uij in paper

                        double e_uij = (prefRelation_uij - normalized_estimate_uij) ;  // from Eq. 3&6
                        double e_uij_derivative = (e_uij * normalized_estimate_uij) / (1 + exp_estimate_uij);
                        
                        // Update feature vectors
                        // Eq. 7
                        Vector<double> P_u = P.Row(indexOfUser);
                        Vector<double> P_u_updated = P_u + learnRate * (
                            (Q.Column(indexOfItem_i) - Q.Column(indexOfItem_j)).Multiply(e_uij_derivative)
                            + P_u.Multiply(regularization));
                        P.SetRow(indexOfUser, P_u_updated);

                        // Eq. 8
                        // TODO: take learnRate * (P_u.Multiply(e_uij_derivative) out for Eq 8&9 to save time
                        Vector<double> Q_i = Q.Column(indexOfItem_i);
                        Vector<double> Q_i_updated = Q_i + (P_u.Multiply(e_uij_derivative) + Q_i.Multiply(regularization)).Multiply(learnRate);
                        Q.SetColumn(indexOfItem_i, Q_i_updated);

                        // Eq. 9
                        Vector<double> Q_j = Q.Column(indexOfItem_j);
                        Vector<double> Q_j_updated = Q_j - (P_u.Multiply(e_uij_derivative) + Q_j.Multiply(regularization)).Multiply(learnRate);
                        Q.SetColumn(indexOfItem_j, Q_j_updated);

                        /*
                        for (int k = 0; k < factorCount; ++k)
                        {
                            double factorOfUser = P[indexOfUser, k];
                            double factorOfItem_i = Q[k, indexOfItem_i];
                            double factorOfItem_j = Q[k, indexOfItem_j];

                            // TODO: Seperate user/item regularization coefficient
                            P[indexOfUser, k] += learnRate * (e_uij * normalized_estimate_uij * factorOfUser - regularization * factorOfUser);
                            // Two items are updated in different directions
                            Q[k, indexOfItem_i] += learnRate * (normalized_estimate_uij * factorOfItem_i - regularization * factorOfItem_i);
                            // Two items are updated in different directions
                            Q[k, indexOfItem_j] -= learnRate * (normalized_estimate_uij * factorOfItem_j - regularization * factorOfItem_j);
                        }
                        */
                    }
                }

                // Display the current regularized error see if it converges
                double previousErrorSum = double.MaxValue;
                double currentErrorSum = 0;
                if (epoch == 0 || epoch == maxEpoch - 1 || epoch % (int)Math.Ceiling(maxEpoch * 0.1) == 4)
                {
                    double eSum = 0;
                    foreach (var pair in PR_train.PreferenceRelationsByUser)
                    {
                        int indexOfUser = pair.Key;
                        SparseMatrix preferenceRelationsOfUser = pair.Value;

                        // For each preference relation of this user, update the latent feature vectors
                        foreach (var entry in preferenceRelationsOfUser.EnumerateIndexed(Zeros.AllowSkip))
                        {
                            int indexOfItem_i = entry.Item1;
                            int indexOfItem_j = entry.Item2;
                            double prefRelation_uij = entry.Item3;

                            // TODO: Maybe it can be faster to do two dot products to remove the substraction (lose sparse  property I think)
                            double estimate_uij = P.Row(indexOfUser).DotProduct(Q.Column(indexOfItem_i) - Q.Column(indexOfItem_j));   // Eq. 2
                            double normalized_estimate_uij = SpecialFunctions.InverseLogit(estimate_uij);   // Eq. 2
                            eSum += Math.Pow((prefRelation_uij - normalized_estimate_uij), 2);  // Sum the error of this preference relation

                            // Sum the regularization term
                            for (int k = 0; k < factorCount; ++k)
                            {
                                eSum += (regularization * 0.5) * (Math.Pow(P[indexOfUser, k], 2)
                                    + Math.Pow(Q[k, indexOfItem_i], 2) + Math.Pow(Q[k, indexOfItem_j], 2));
                            }
                        }
                    }
                    // Record the current error
                    currentErrorSum = eSum;

                    Utils.PrintEpoch("Epoch", epoch, maxEpoch, "Learning error", Math.Sqrt(eSum / prefCount));
                }

                // Stop the learning if the regularized error falls below a certain threshold
                if (previousErrorSum - currentErrorSum < 0.0001)
                {
                    Console.WriteLine("Improvment less than 0.0001, learning stopped.");
                    break;
                }
                previousErrorSum = currentErrorSum;
            }
            return new RatingMatrix(R_unknown.Matrix.PointwiseMultiply(P.Multiply(Q)));
        }
    }
}
