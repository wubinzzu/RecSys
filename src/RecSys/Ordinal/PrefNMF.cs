using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Core;
using RecSys.Numerical;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Linq;

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
        public static DataMatrix PredictRatings(PrefRelations PR_train, DataMatrix R_unknown,
           int maxEpoch, double learnRate, double regularizationOfUser, double regularizationOfItem, int factorCount)
        {
            // Latent features
            List<Vector<double>> P;
            List<Vector<double>> Q;

            LearnLatentFeatures(PR_train, maxEpoch, learnRate, regularizationOfUser, regularizationOfItem, factorCount, out P, out Q);

            List<Tuple<int, int, double>> R_predicted_cache = new List<Tuple<int, int, double>>();
            foreach(var data in R_unknown.Matrix.EnumerateIndexed(Zeros.AllowSkip))
            {
                int indexOfUser = data.Item1;
                int indexOfItem = data.Item2;
                R_predicted_cache.Add(new Tuple<int, int, double>(indexOfUser, indexOfItem, P[indexOfUser].DotProduct(Q[indexOfItem])));
            }

            DataMatrix R_predicted = new DataMatrix(SparseMatrix.OfIndexed(R_unknown.UserCount,R_unknown.ItemCount,R_predicted_cache));
                //new DataMatrix(R_unknown.Matrix.PointwiseMultiply(P.Multiply(Q)));
            // TODO: should we do this? should we put it into [0,1]??? Seems zero entries are also converted into 0.5!Normalize the result
            //R_predicted.Matrix.MapInplace(x => RecSys.Core.SpecialFunctions.InverseLogit(x), Zeros.AllowSkip);
            return R_predicted;
        }

        // We need to directly compute the position matrix because the PR would be too big to fit into memory
        public static SparseMatrix PredictPrefRelations(PrefRelations PR_train, Dictionary<int, List<int>> PR_unknown,
    int maxEpoch, double learnRate, double regularizationOfUser, double regularizationOfItem, int factorCount, List<double> quantizer)
        {
            // Latent features
            List<Vector<double>> P;
            List<Vector<double>> Q;
            //Matrix<double> P;
            //Matrix<double> Q;


            //SparseMatrix positionMatrix = new SparseMatrix(PR_train.UserCount, PR_train.ItemCount);
            Vector<double>[] positionMatrixCache = new Vector<double>[PR_train.UserCount];
            LearnLatentFeatures(PR_train, maxEpoch, learnRate, regularizationOfUser, regularizationOfItem, factorCount, out P, out Q);

            //PrefRelations PR_predicted = new PrefRelations(PR_train.ItemCount);

            Object lockMe = new Object();
            Parallel.ForEach(PR_unknown, user =>
            {
                Utils.PrintEpoch("Epoch", user.Key, PR_unknown.Count);
                int indexOfUser = user.Key;
                List<int> unknownItemsOfUser = user.Value;
                //SparseMatrix predictedPreferencesOfUser = new SparseMatrix(PR_train.ItemCount, PR_train.ItemCount);
                List<Tuple<int, int, double>> predictedPreferencesOfUserCache = new List<Tuple<int, int, double>>();

                // Predict each unknown preference
                foreach (int indexOfItem_i in unknownItemsOfUser)
                {
                    foreach (int indexOfItem_j in unknownItemsOfUser)
                    {
                        if (indexOfItem_i == indexOfItem_j) continue;
                        double estimate_uij = P[indexOfUser].DotProduct(Q[indexOfItem_i] - Q[indexOfItem_j]);   // Eq. 2
                        double normalized_estimate_uij = Core.SpecialFunctions.InverseLogit(estimate_uij);   // pi_uij in paper
                        predictedPreferencesOfUserCache.Add(new Tuple<int, int, double>(indexOfItem_i, indexOfItem_j, normalized_estimate_uij));
                        //predictedPreferencesOfUser[indexOfItem_i, indexOfItem_j] = normalized_estimate_uij;
                    }
                }

                // Note: it shows better performance to not quantize here
                /*
                DataMatrix predictedPreferencesOfUser = 
                    new DataMatrix(SparseMatrix.OfIndexed(PR_train.ItemCount, PR_train.ItemCount, predictedPreferencesOfUserCache));
                predictedPreferencesOfUser.Quantization(0, 1.0, quantizer);    
                Vector<double> positionsOfUser = PrefRelations.PreferencesToPositions(predictedPreferencesOfUser.Matrix);
                */
                
                double[] positionByItem = new double[PR_train.ItemCount];
                foreach(var triplet in predictedPreferencesOfUserCache)
                {
                    int indexOfItem_i = triplet.Item1;
                    int indexOfItem_j = triplet.Item2;
                    double preference = triplet.Item3;
                    if(preference > 0.5)
                    {
                        positionByItem[indexOfItem_i]++;
                        positionByItem[indexOfItem_j]--;
                    }
                    else if(preference < 0.5)
                    {
                        positionByItem[indexOfItem_i]--;
                        positionByItem[indexOfItem_j]++;
                    }
                }

                int normalizationTerm = unknownItemsOfUser.Count * 2 - 2;
                for (int i = 0; i < positionByItem.Length; i ++ )
                {
                    if (positionByItem[i]!=0)
                        positionByItem[i] /= normalizationTerm;
                }
                
                Vector<double> positionsOfUser = SparseVector.OfEnumerable(positionByItem);
                
                lock (lockMe)
                {
                    positionMatrixCache[indexOfUser] = positionsOfUser;
                    //positionMatrix.SetRow(indexOfUser, positionsOfUser);
                    //PR_predicted[indexOfUser] = predictedPreferencesOfUser;
                }
            });

            return SparseMatrix.OfRowVectors(positionMatrixCache);
        }


        public static PrefRelations PredictPrefRelations(PrefRelations PR_train, SparseMatrix PR_unknown,
            int maxEpoch, double learnRate, double regularizationOfUser, double regularizationOfItem, int factorCount)
        {
            // Latent features
            List<Vector<double>> P;
            List<Vector<double>> Q;
            //Matrix<double> P;
            //Matrix<double> Q;

            LearnLatentFeatures(PR_train, maxEpoch, learnRate, regularizationOfUser, regularizationOfItem, factorCount, out P, out Q);

            PrefRelations PR_predicted = new PrefRelations(PR_train.ItemCount);

            Object lockMe = new Object();
            Parallel.ForEach(PR_unknown.EnumerateRowsIndexed(), user =>
            {
                int indexOfUser = user.Item1;
                Vector<double> unknownPreferencesOfUser = user.Item2;
                SparseMatrix predictedPreferencesOfUser = new SparseMatrix(PR_train.ItemCount, PR_train.ItemCount);

                // Predict each unknown preference
                foreach(var unknownPreference in unknownPreferencesOfUser.EnumerateIndexed(Zeros.AllowSkip))
                {
                    int indexOfItem_i = unknownPreference.Item1;
                    int indexOfItem_j = (int)unknownPreference.Item2;
                    double estimate_uij = P[indexOfUser].DotProduct(Q[indexOfItem_i] - Q[indexOfItem_j]);   // Eq. 2
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

        private static void LearnLatentFeatures(PrefRelations PR_train, int maxEpoch, 
            double learnRate, double regularizationOfUser, double regularizationOfItem,
            int factorCount, out List<Vector<double>> P, out List<Vector<double>> Q)
        {
            //regularizationOfUser = 0;
            //regularizationOfItem = 0;
            int userCount = PR_train.UserCount;
            int itemCount = PR_train.ItemCount;

            // User latent vectors with default seed
            P = new List<Vector<double>>();
            Q = new List<Vector<double>>();
            ContinuousUniform uniformDistribution = new ContinuousUniform(0, 0.1, new Random(Config.Seed));
            //var p = Utils.CreateRandomMatrixFromUniform(userCount, factorCount, 0, 0.1, Config.Seed);
            for (int i = 0; i < userCount; i++ )
            {
                P.Add(DenseVector.CreateRandom(factorCount,uniformDistribution));
            }
            for (int i = 0; i < itemCount; i++)
            {
                Q.Add(DenseVector.CreateRandom(factorCount, uniformDistribution));
            }
             //   P = Utils.CreateRandomMatrixFromUniform(userCount, factorCount, 0, 0.1, Config.Seed);
            // Item latent vectors with a different seed
            //Q = Utils.CreateRandomMatrixFromUniform(factorCount, itemCount, 0, 0.1, Config.Seed + 1);

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
                        double PQ_ui = P[indexOfUser].DotProduct(Q[indexOfItem_i]);
                        double PQ_uj = P[indexOfUser].DotProduct(Q[indexOfItem_j]);
                        double estimate_uij = PQ_ui - PQ_uj;
                        //double estimate_uij = P.Row(indexOfUser).DotProduct(Q.Column(indexOfItem_i) - Q.Column(indexOfItem_j));   // Eq. 2
                        
                        
                        double exp_estimate_uij = Math.Exp(estimate_uij);   // enumerator in Eq. 2
                        double normalized_estimate_uij = SpecialFunctions.InverseLogit(estimate_uij);   // pi_uij in paper
                        
                        //Debug.Assert(prefRelation_uij >= 0 && prefRelation_uij <= 1);
                        //Debug.Assert(normalized_estimate_uij >= 0 && normalized_estimate_uij <= 1);


                        // The error term in Eq. 6-9. Note that the author's paper incorrectly puts a power on the error
                        double e_uij = prefRelation_uij - normalized_estimate_uij;
                        //double e_uij = Math.Pow(prefRelation_uij - normalized_estimate_uij, 2) ;  // from Eq. 3&6
                        double e_uij_derivative = (e_uij * normalized_estimate_uij) / (1 + exp_estimate_uij);

                        // Update feature vectors
                        Vector<double> P_u = P[indexOfUser];
                        Vector<double> Q_i = Q[indexOfItem_i];
                        Vector<double> Q_j = Q[indexOfItem_j];
                        Vector<double> Q_ij = Q_i - Q_j;
   
                        P[indexOfUser] += Q_ij.Multiply(e_uij_derivative * learnRate) - P_u.Multiply(regularizationOfUser * learnRate);

                        // Eq. 7, note that the author's paper incorrectly writes + regularization 
                        //Vector<double> P_u_updated = P_u + (Q_ij.Multiply(e_uij_derivative) - P_u.Multiply(regularizationOfUser)).Multiply(learnRate);
                        //P[indexOfUser] = P_u_updated;
                        Vector<double> P_u_derivative = P_u.Multiply(e_uij_derivative * learnRate);
                        // Eq. 8, note that the author's paper incorrectly writes + regularization 
                        //Vector<double> Q_i_updated = Q_i + (P_u_derivative - Q_i.Multiply(regularizationOfItem * learnRate));
                        //Q[indexOfItem_i] = Q_i_updated;

                        Q[indexOfItem_i] += (P_u_derivative - Q_i.Multiply(regularizationOfItem * learnRate));

                        // Eq. 9, note that the author's paper incorrectly writes + regularization 
                        //Vector<double> Q_j_updated = Q_j - (P_u_derivative - Q_j.Multiply(regularizationOfItem * learnRate));
                        //Q[indexOfItem_j] =Q_j_updated;
                        Q[indexOfItem_j] -= (P_u_derivative - Q_j.Multiply(regularizationOfItem * learnRate));

                        double estimate_uij_updated = P[indexOfUser].DotProduct(Q[indexOfItem_i] - Q[indexOfItem_j]);   // Eq. 2
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
                            double estimate_uij = P[indexOfUser].DotProduct(Q[indexOfItem_i] - Q[indexOfItem_j]);   // Eq. 2
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
                    double regularizationPenaty = regularizationOfUser * P.Sum(x=>x.SquaredSum());
                    regularizationPenaty += regularizationOfItem * Q.Sum(x => x.SquaredSum());
                    eSum += regularizationPenaty;

                    // Record the current error
                    currentErrorSum = eSum;

                    Utils.PrintEpoch("Epoch", epoch, maxEpoch, "Learning error", eSum.ToString("0.0"), true);
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
