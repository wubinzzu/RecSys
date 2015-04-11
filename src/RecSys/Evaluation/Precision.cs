using System.Collections.Generic;

namespace RecSys
{
    public static class Precision
    {
        public static double Evaluate(Dictionary<int, List<int>> correctLists,
            Dictionary<int, List<int>> predictedLists, int topN)
        {
            int countOfTests = 0;
            double precisionSum = 0;

            foreach (int user in correctLists.Keys)
            {
                double precision = PrecisionAtTopN(correctLists[user], predictedLists[user], topN);

                // If there are fewer relevant items than N then this user 
                // will be skipped from evaluation this is the same as consider 
                // this user has never been included in test set
                // let me know if there is a better treatment
                if (precision != -1)
                {
                    precisionSum += precision;
                    countOfTests++;
                }
            }
            return precisionSum / countOfTests;
        }

        private static double PrecisionAtTopN(List<int> correctList, List<int> predictedList, int topN)
        {
            if (correctList.Count < topN)
            {
                return -1;  // Invalid test
            }

            int hits = 0;
            for (int i = 0; i < topN && i < predictedList.Count; i++)
            {
                int indexOfItem = predictedList[i];
                if (correctList.Contains(indexOfItem)) { hits++; }
            }

            return (double)hits / topN;
        }
    }
}
