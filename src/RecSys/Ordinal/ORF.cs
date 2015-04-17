using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using RecSys.Numerical;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Ordinal
{
    class ORF
    {
        Vector<double> meanByUser;
        Vector<double> meanByItem;
        RatingMatrix R_train;
        SparseMatrix similarityByItemItem;
        // The weights for item-item correlation features
        // It is the \vec{w} in the paper, and featureWeightByItemItem[i,j] is w_ij
        SparseMatrix featureWeightByItemItem;
        Dictionary<Tuple<int, int>, double[]> OMFDistributions;

        public void PredictRatings(RatingMatrix R_train, RatingMatrix R_unknown, Matrix<double> fullSimilarityByItemItem, Dictionary<Tuple<int, int>, double[]> OMFDistributions, double regularization, double learnRate, double minSimilarity, int maxEpoch, int ratingLevels, out RatingMatrix R_predicted_expectations, out RatingMatrix R_predicted_mostlikely)
        {
            /************************************************************
             *   Parameterization and Initialization
            ************************************************************/
            #region Parameterization and Initialization
            int userCount = R_train.UserCount;
            int itemCount = R_train.ItemCount;
            meanByUser = R_train.GetUserMeans(); // Mean value of each user
            meanByItem = R_train.GetItemMeans(); // Mean value of each item
            this.R_train = R_train;
            this.OMFDistributions = OMFDistributions;
            R_predicted_expectations = new RatingMatrix(R_unknown.UserCount, R_unknown.ItemCount);
            R_predicted_mostlikely = new RatingMatrix(R_unknown.UserCount, R_unknown.ItemCount);

            featureWeightByItemItem = new SparseMatrix(itemCount); 

            // Initialize the weights
            // Remove weak similarities so that the # of features is limited, see paper for more details
            fullSimilarityByItemItem.CoerceZero(minSimilarity);
            similarityByItemItem = SparseMatrix.OfMatrix(fullSimilarityByItemItem);    // After removed weak similarities the matrix should be sparse
            // Initialize all strong item-item features
            Random rnd = new Random(Config.Seed);
            foreach(var element in similarityByItemItem.EnumerateIndexed(Zeros.AllowSkip))
            {
                int indexOfItem_i = element.Item1;
                int indexOfItem_j = element.Item2;
                double randomWeight = rnd.NextDouble() * 0.01;
                featureWeightByItemItem[indexOfItem_i, indexOfItem_j] = randomWeight;
                featureWeightByItemItem[indexOfItem_j, indexOfItem_i] = randomWeight;
            }
            Debug.Assert(similarityByItemItem.NonZerosCount == featureWeightByItemItem.NonZerosCount);

            // We cache here which items have been rated by the given user
            // it will be reused in every feature update
            Dictionary<int, List<int>> itemsByUser = R_train.GetItemsByUser();

            // TODO: we actually stored more features, because some items may not be co-rated by any user
            Utils.PrintValue("# of item-item features", (featureWeightByItemItem.NonZerosCount / 2).ToString());

            #endregion

            /************************************************************
             *   Learn weights from training data R_train
            ************************************************************/
            #region Learn weights from training data R_train
            for (int epoch = 0; epoch < maxEpoch; epoch++)
            {
                double likelihood = 0;
                // Train with batch ratings by each user
                foreach(var user in R_train.Users)
                {
                    int indexOfUser = user.Item1;
                    Vector<double> ratingsOfUser = user.Item2;
                    Debug.Assert(ratingsOfUser.Storage.IsDense == false, "The user ratings should be stored in a sparse vector.");

                    List<int> itemsOfUser = itemsByUser[indexOfUser];   // Cache the items rated by this user

                    // Learn with each pair of items co-rated by this user
                    foreach (var item_i in ratingsOfUser.EnumerateIndexed(Zeros.AllowSkip))
                    {
                        int indexOfItem_i = item_i.Item1;
                        foreach (var item_j in ratingsOfUser.EnumerateIndexed(Zeros.AllowSkip))
                        {
                            #region Preparation
                            int indexOfItem_j = item_j.Item1;

                            // As i-j and j-i correspond to the same feature, 
                            // so we train only i < j to avoid double training
                            if (indexOfItem_i > indexOfItem_j) { continue; }

                            // If the similarity is zero then it is a weak feature and we skip it
                            // recall that we have set weak similarity to zero
                            if (similarityByItemItem[indexOfItem_i, indexOfItem_j] == SparseMatrix.Zero) { continue; }

                            // Otherwise i-j is a strong feature and we will update its weight
                            // Find out the neighbors of item_i where the 
                            List<int> neighborsOfItem_i = new List<int>(itemsOfUser);
                            neighborsOfItem_i.Remove(indexOfItem_i);    // It is not a neighbor of itself
                            // Remove weak neighbors
                            foreach (int indexOfNeighbor in itemsOfUser)
                            {
                                if (similarityByItemItem[indexOfItem_i, indexOfNeighbor] == SparseMatrix.Zero)
                                {
                                    neighborsOfItem_i.Remove(indexOfNeighbor);
                                }
                            }
                            #endregion

                            // ================Compute gradients===================
                            #region Compute gradients
                            double r_ui = R_train[indexOfUser, indexOfItem_i];
                            double r_uj = R_train[indexOfUser, indexOfItem_j];
                            double meanOfItem_i = meanByItem[indexOfItem_i];
                            double meanOfItem_j = meanByItem[indexOfItem_j];
                            double localLikelihood_ui = 0.0;

                            // Partition function Z
                            double Z_ui = 0;
                            for (int targetRating = 1; targetRating <= ratingLevels; targetRating++)
                            {
                                Z_ui += OMFDistributions[new Tuple<int, int>(indexOfUser, indexOfItem_i)][targetRating - 1] * ComputePotential(targetRating, indexOfUser, indexOfItem_i, neighborsOfItem_i);
                            }

                            // Compute the first term
                            double gradientFirstTerm = ComputeCorrelationFeature(r_ui, meanOfItem_i, r_uj, meanOfItem_j);

                            // Compute the second term
                            double gradientSecondTerm = 0.0;
                            for (int targetRating = 1; targetRating <= ratingLevels; targetRating++)
                            {
                                double localLikelihood = ComputeLocalLikelihood(targetRating, indexOfUser, indexOfItem_i, neighborsOfItem_i, Z_ui);
                                double correlationFeature = ComputeCorrelationFeature(targetRating, meanOfItem_i, r_uj, meanOfItem_j);
                                gradientSecondTerm += localLikelihood * correlationFeature;

                                //TODO: Not sure
                                if (targetRating == r_ui)
                                    localLikelihood_ui = localLikelihood;
                            }

                            // Merge all terms
                            double gradient = gradientFirstTerm - gradientSecondTerm;

                            #endregion

                            #region Update weights
                            // Add regularization term
                            double weight = featureWeightByItemItem[indexOfItem_i, indexOfItem_j];
                            localLikelihood_ui -= (weight * weight) / (2 * regularization * regularization);
                            gradient -= weight / regularization;

                            likelihood += Math.Log(localLikelihood_ui);

                            // Update the weight with gradient
                            double step = learnRate * gradient;
                            featureWeightByItemItem[indexOfItem_i, indexOfItem_j] += step;

                            // The weights are mirrored
                            featureWeightByItemItem[indexOfItem_j, indexOfItem_i] += step;

                            Debug.Assert(featureWeightByItemItem[indexOfItem_i, indexOfItem_j] == featureWeightByItemItem[indexOfItem_j, indexOfItem_i]);

                            #endregion
                        }
                    }
                }
            }
            #endregion

            /************************************************************
             *   Make predictions
            ************************************************************/
            #region Make predictions

            foreach(var user in R_unknown.Users)
            {
                int indexOfUser = user.Item1;
                Vector<double> unknownRatingsOfUser = user.Item2;
                List<int> itemsOfUser = itemsByUser[indexOfUser];

                foreach(var unknownRating in unknownRatingsOfUser.EnumerateIndexed(Zeros.AllowSkip))
                {
                    int indexOfItem = unknownRating.Item1;

                    List<int> neighborsOfItem = new List<int>(itemsOfUser);
                    neighborsOfItem.Remove(indexOfItem);    // It is not a neighbor of itself
                    // Remove weak neighbors
                    foreach (int indexOfNeighbor in itemsOfUser)
                    {
                        if (similarityByItemItem[indexOfItem, indexOfNeighbor] == SparseMatrix.Zero)
                        {
                            neighborsOfItem.Remove(indexOfNeighbor);
                        }
                    }

                    // Partition function Z
                    double Z_ui = 0;
                    for (int targetRating = 1; targetRating <= ratingLevels; targetRating++)
                    {
                        Z_ui += OMFDistributions[new Tuple<int, int>(indexOfUser, indexOfItem)][targetRating - 1] * ComputePotential(targetRating, indexOfUser, indexOfItem, neighborsOfItem);
                    }

                    double sumOfLikelihood = 0.0;
                    double currentMaxLikelihood = 0.0;
                    double mostlikelyRating = 0.0;
                    double expectationRating = 0.0;
                    for (int targetRating = 1; targetRating <= ratingLevels; targetRating++)
                    {
                        double likelihoodOfTargetRating = ComputeLocalLikelihood(targetRating, indexOfUser, indexOfItem, neighborsOfItem, Z_ui);

                        // Compute the most likely rating for MAE
                        if (likelihoodOfTargetRating > currentMaxLikelihood)
                        {
                            mostlikelyRating = targetRating;
                            currentMaxLikelihood = likelihoodOfTargetRating;
                        }

                        // Compute expectation for RMSE
                        expectationRating += targetRating * likelihoodOfTargetRating;

                        sumOfLikelihood += likelihoodOfTargetRating;
                    }

                    // The sum of likelihoods should be 1, maybe not that high precision though
                    Debug.Assert(Math.Abs(sumOfLikelihood - 1.0) < 0.0001);

                    R_predicted_expectations[indexOfUser, indexOfItem] = expectationRating;
                    R_predicted_mostlikely[indexOfUser, indexOfItem] = mostlikelyRating;
                }
            }

            #endregion

        }

        #region Compute the potentials
        private double ComputePotential(double targetRating, int indexOfUser, int indexOfItem_i, List<int> neighborsOfItem_i)
        {
            // Correlation potential
            double totalCorrelationPotential = 1;
            foreach (int indexOfNeighbor in neighborsOfItem_i)
            {
                double correlationFeature = ComputeCorrelationFeature(targetRating, meanByItem[indexOfItem_i],
                    R_train[indexOfUser, indexOfNeighbor], meanByItem[indexOfNeighbor]);

                double strength = similarityByItemItem[indexOfItem_i, indexOfNeighbor];

                double weight = featureWeightByItemItem[indexOfItem_i, indexOfNeighbor];

                // We should not have 0 weight for two reasons:
                // zero weight means it never get initialized, which means there is 
                // no edge (two items rated by the same user) between the corresponding
                // items. However, we do have a very rare chance that the weight happended
                // to be randomly assigned/updated to 0
                Debug.Assert(weight != 0);

                totalCorrelationPotential *= Math.Exp(correlationFeature * weight);
            }

            return totalCorrelationPotential;
        }

        #endregion

        #region Compute correlation features
        private double ComputeCorrelationFeature(double r_ui, double ave_i, double r_uj, double ave_j)
        {
            double feature_ij = NormalizeFeatureValue(Math.Abs((r_ui - ave_i) - (r_uj - ave_j)));
            Debug.Assert(feature_ij >= 0 && feature_ij <= 1);
            return feature_ij;
        }
        #endregion

        // TODO: The normalizatio is not very correct,
        // should normalize into [0,1] but it does [0.5,1]
        #region Normalize feature values
        private double NormalizeFeatureValue(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }
        #endregion

        #region Compute local likelihood
        private double ComputeLocalLikelihood(int targetRating, int indexOfUser, int indexOfItem, List<int> neighborsOfItem, double Z_ui)
        {
            double potential = ComputePotential(targetRating, indexOfUser, indexOfItem, neighborsOfItem);
            double numerator = OMFDistributions[new Tuple<int,int>(indexOfUser,indexOfItem)][targetRating-1]* potential;

            // TODO: Sth wrong with the numerator? NaN?
            if (numerator > Z_ui)
            {
                Debug.Assert(numerator <= Z_ui);
            }

            return numerator / Z_ui;
        }
        #endregion
    }
}
