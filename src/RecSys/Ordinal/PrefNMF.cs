using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Core;
using RecSys.Numerical;
using System;
using System.Diagnostics;
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
           int maxEpoch, double learnRate, double regularizationOfUser, double regularizationOfItem, int factorCount)
        {
            // Latent features
            Matrix<double> P;
            Matrix<double> Q;

            LearnLatentFeatures(PR_train, maxEpoch, learnRate, regularizationOfUser, regularizationOfItem, factorCount, out P, out Q);

            RatingMatrix R_predicted = new RatingMatrix(R_unknown.Matrix.PointwiseMultiply(P.Multiply(Q)));
            // TODO: should we do this? should we put it into [0,1]??? Seems zero entries are also converted into 0.5!Normalize the result
            //R_predicted.Matrix.MapInplace(x => RecSys.Core.SpecialFunctions.InverseLogit(x), Zeros.AllowSkip);
            return R_predicted;
        }

        public static PrefRelations PredictPrefRelations(PrefRelations PR_train, PrefRelations PR_unknown,
            int maxEpoch, double learnRate, double regularizationOfUser, double regularizationOfItem, int factorCount)
        {
            // Latent features
            Matrix<double> P;
            Matrix<double> Q;

            LearnLatentFeatures(PR_train, maxEpoch, learnRate, regularizationOfUser, regularizationOfItem, factorCount, out P, out Q);

            PrefRelations PR_predicted = new PrefRelations(PR_unknown.ItemCount);

            Object lockMe = new Object();
            Parallel.ForEach(PR_unknown.PreferenceRelationsByUser, pair =>
            {
                int indexOfUser = pair.Key;
                SparseMatrix unknownPreferencesOfUser = pair.Value;
                SparseMatrix predictedPreferencesOfUser = new SparseMatrix(unknownPreferencesOfUser.RowCount, unknownPreferencesOfUser.ColumnCount);

                // Predict each unknown preference
                foreach(var unknownPreference in unknownPreferencesOfUser.EnumerateIndexed(Zeros.AllowSkip))
                {
                    int indexOfItem_i = unknownPreference.Item1;
                    int indexOfItem_j = unknownPreference.Item2;
                    double estimate_uij = P.Row(indexOfUser).DotProduct(Q.Column(indexOfItem_i) - Q.Column(indexOfItem_j));   // Eq. 2
                    double normalized_estimate_uij = Core.SpecialFunctions.InverseLogit(estimate_uij);   // pi_uij in paper
                    predictedPreferencesOfUser[indexOfItem_i, indexOfItem_j] = normalized_estimate_uij;
                }

                lock(lockMe)
                {
                    PR_predicted[indexOfUser] = predictedPreferencesOfUser;
                }
            });

            return PR_predicted;
        }

        private static void LearnLatentFeatures(PrefRelations PR_train, int maxEpoch, double learnRate, double regularizationOfUser, double regularizationOfItem, int factorCount, out Matrix<double> P, out Matrix<double> Q)
        {
            //regularizationOfUser = 0;
            //regularizationOfItem = 0;
            int userCount = PR_train.UserCount;
            int itemCount = PR_train.ItemCount;
            int prefCount = PR_train.GetTotalPrefRelationsCount();

            // User latent vectors with default seed
            P = Utils.CreateRandomMatrix(userCount, factorCount, Config.Seed);
            // Item latent vectors with a different seed
            Q = Utils.CreateRandomMatrix(factorCount, itemCount, Config.Seed + 1);

            // SGD
            double previousErrorSum = long.MaxValue;
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
                        //Console.WriteLine(preferenceRelationsOfUser[indexOfItem_i, indexOfItem_j]);
                        //Console.WriteLine(preferenceRelationsOfUser[indexOfItem_j, indexOfItem_i]);
                        if (indexOfItem_i >= indexOfItem_j) continue;

                        // Warning: here we need to convert the customized preference indicators
                        // from 1,2,3 into 0,0.5,1 for match the scale of predicted pi, which is in range [0,1] 
                        double prefRelation_uij = 0;
                        if(entry.Item3 == Config.Preferences.Preferred){prefRelation_uij = 1.0;}
                        else if (entry.Item3 == Config.Preferences.EquallyPreferred){prefRelation_uij = 0.5;}
                        else if (entry.Item3 == Config.Preferences.LessPreferred){prefRelation_uij = 0.0;}
                        else{Debug.Assert(true, "Should not be here.");}

                        // TODO: Maybe it can be faster to do two dot products to remove the substraction (lose sparse  property I think)
                        double estimate_uij = P.Row(indexOfUser).DotProduct(Q.Column(indexOfItem_i) - Q.Column(indexOfItem_j));   // Eq. 2
                        double exp_estimate_uij = Math.Exp(estimate_uij);   // enumerator in Eq. 2
                        double normalized_estimate_uij = SpecialFunctions.InverseLogit(estimate_uij);   // pi_uij in paper

                        //Debug.Assert(prefRelation_uij >= 0 && prefRelation_uij <= 1);
                        //Debug.Assert(normalized_estimate_uij >= 0 && normalized_estimate_uij <= 1);


                        // TODO: try square the e_uij
                        // I think it should be squared, when sqaured and set regularization to 0,
                        // the updated feature vectors actually increase the error!
                        // where it won't happen without square and also for NMF
                        // squared is like  always gradient in one direction
                        double e_uij = prefRelation_uij - normalized_estimate_uij;
                        //double e_uij = Math.Pow(prefRelation_uij - normalized_estimate_uij, 2) ;  // from Eq. 3&6
                        double e_uij_derivative = (e_uij * normalized_estimate_uij) / (1 + exp_estimate_uij);

                        // Update feature vectors
                        Vector<double> P_u = P.Row(indexOfUser);
                        Vector<double> Q_i = Q.Column(indexOfItem_i);
                        Vector<double> Q_j = Q.Column(indexOfItem_j);
                        Vector<double> Q_ij = Q_i - Q_j;
                        // Eq. 7
                        Vector<double> P_u_updated = P_u + (Q_ij.Multiply(e_uij_derivative) + P_u.Multiply(regularizationOfUser)).Multiply(learnRate);
                        P.SetRow(indexOfUser, P_u_updated);

                        // Eq. 8
                        Vector<double> Q_i_updated = Q_i + (P_u.Multiply(e_uij_derivative) + Q_i.Multiply(regularizationOfItem)).Multiply(learnRate);
                        Q.SetColumn(indexOfItem_i, Q_i_updated);

                        // Eq. 9, note that changing the minus to plus will increase error
                        Vector<double> Q_j_updated = Q_j - (P_u.Multiply(e_uij_derivative) + Q_j.Multiply(regularizationOfItem)).Multiply(learnRate);
                        Q.SetColumn(indexOfItem_j, Q_j_updated);

                        double estimate_uij_updated = P.Row(indexOfUser).DotProduct(Q.Column(indexOfItem_i) - Q.Column(indexOfItem_j));   // Eq. 2
                        double exp_estimate_uij_updated = Math.Exp(estimate_uij_updated);   // enumerator in Eq. 2
                        double normalized_estimate_uij_updated = SpecialFunctions.InverseLogit(estimate_uij_updated);   // pi_uij in paper
                        //double e_uij_updated = Math.Pow(prefRelation_uij - normalized_estimate_uij_updated, 2);  // from Eq. 3&6
                        double e_uij_updated = prefRelation_uij - normalized_estimate_uij_updated;  // from Eq. 3&6

                        //double debug1 = Math.Abs(e_uij) - Math.Abs(e_uij_updated);
                        // Debug.Assert(debug1 > 0);    // After update the error should be smaller

                        #region Loop version of gradient update
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
                        #endregion
                    }
                }

                // Display the current regularized error see if it converges
                double currentErrorSum = 0;
                //if (epoch == 0 || epoch == maxEpoch - 1 || epoch % (int)Math.Ceiling(maxEpoch * 0.1) == 4)
                if (true)
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

                            if (indexOfItem_i >= indexOfItem_j) continue;

                            double prefRelation_uij = 0;
                            if (entry.Item3 == Config.Preferences.Preferred) { prefRelation_uij = 1.0; }
                            else if (entry.Item3 == Config.Preferences.EquallyPreferred) { prefRelation_uij = 0.5; }
                            else if (entry.Item3 == Config.Preferences.LessPreferred) { prefRelation_uij = 0.0; }
                            else { Debug.Assert(true, "Should not be here."); }

                            // TODO: Maybe it can be faster to do two dot products to remove the substraction (lose sparse  property I think)
                            double estimate_uij = P.Row(indexOfUser).DotProduct(Q.Column(indexOfItem_i) - Q.Column(indexOfItem_j));   // Eq. 2
                            double normalized_estimate_uij = SpecialFunctions.InverseLogit(estimate_uij);   // Eq. 2
                            eSum += Math.Pow((prefRelation_uij - normalized_estimate_uij), 2);  // Sum the error of this preference relation

                            // Sum the regularization term
                            //for (int k = 0; k < factorCount; ++k)
                            // {
                            //     eSum += (regularizationOfUser * 0.5) * (Math.Pow(P[indexOfUser, k], 2)
                            //         + Math.Pow(Q[k, indexOfItem_i], 2) + Math.Pow(Q[k, indexOfItem_j], 2));
                            // }
                        }
                    }
                    double regularizationPenaty = regularizationOfUser * P.SquaredSum();
                    regularizationPenaty += regularizationOfItem * Q.SquaredSum();
                    eSum += regularizationPenaty;

                    // Record the current error
                    currentErrorSum = eSum;

                    Utils.PrintEpoch("Epoch", epoch, maxEpoch, "Learning error", Math.Sqrt(eSum / prefCount));
                    Utils.PrintValue("" + epoch + "/" + maxEpoch, eSum.ToString("0.0"));
                    // Stop the learning if the regularized error falls below a certain threshold
                    // Actually we only check it once every several epoches
                    if (previousErrorSum - currentErrorSum < 0.0001)
                    {
                        Console.WriteLine("Improvment less than 0.0001, learning stopped.");
                        break;
                    }
                    previousErrorSum = currentErrorSum;
                }
            }

        }
    }
}
