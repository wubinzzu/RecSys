using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Evaluation
{
    /// <summary>
    /// https://www.kaggle.com/wiki/MeanAveragePrecision
    /// https://www.youtube.com/watch?v=pM6DJ0ZZee0
    /// </summary>
    class MAP
    {
       public static double Evaluate(Dictionary<int, List<int>> correctLists,
            Dictionary<int, List<int>> predictedLists, int topN)
        {
            double apSum = 0;

           foreach(var prediction  in predictedLists)
           {
               int indexOfUser = prediction.Key;
               List<int> predictedList = prediction.Value;
               List<int> correctList = correctLists[indexOfUser];

               if(predictedList.Count==0 || correctList.Count==0)
               {
                   continue;   // the average precision is zero if either list is empty
               }

               List<int> predictedListTopN = predictedList.GetRange(0, Math.Min(topN, predictedList.Count));
               double hits = 0;
               double ap = 0;
               for(int i = 0 ; i < predictedListTopN.Count;i++)
               {
                   int rank = i + 1;
                   int indexOfItem = predictedListTopN[i];
                   if (correctList.Contains(indexOfItem))
                   {
                       hits++;
                       ap += hits / rank;
                   }
               }
               ap /= Math.Min(correctList.Count,topN);
               apSum += ap;
           }

           return apSum / predictedLists.Count;
        }
    }
}
