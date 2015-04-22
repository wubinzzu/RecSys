using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Core
{
    /// <summary>
    /// Due to large datasets, we need to design a structure
    /// to store and retrieve similarity data efficiently
    /// </summary>
    [Serializable]
    public class SimilarityData
    {
        public SimilarityData() { }
        public Dictionary<int, List<KeyValuePair<int, double>>> neighborsByObject;
        public Dictionary<int, bool> sortedStatuByObject;
        public int BufferSize { get; set; }   // We will force a sort when there are too many neighbors
        public int MaxCountOfNeighbors { get; set; }

        // Sort and remove neighbors of an object
        private void SortAndRemoveNeighbors(int indexOfObject)
        {
            int countOfNeighbor = neighborsByObject[indexOfObject].Count;
            neighborsByObject[indexOfObject].Sort((firstPair, nextPair) =>
            {
                return nextPair.Value.CompareTo(firstPair.Value);
                //return firstPair.Value.CompareTo(nextPair.Value);
            });

            neighborsByObject[indexOfObject] = neighborsByObject[indexOfObject]
                .GetRange(0, countOfNeighbor < MaxCountOfNeighbors ? countOfNeighbor:MaxCountOfNeighbors);

            sortedStatuByObject[indexOfObject] = true;

            Debug.Assert(neighborsByObject[indexOfObject].Count <= MaxCountOfNeighbors);
        }

        public void SortAndRemoveNeighbors()
        {
            List<int> indexesOfObject = new List<int>(neighborsByObject.Keys);

            foreach(int indexOfObject in indexesOfObject)
            {
                if (sortedStatuByObject[indexOfObject] == false)
                {
                    SortAndRemoveNeighbors(indexOfObject);
                }
            }
        }

        public SimilarityData(int maxCountOfNeighbors, int bufferSize = 500)
        {
            MaxCountOfNeighbors = maxCountOfNeighbors;
            neighborsByObject = new Dictionary<int, List<KeyValuePair<int, double>>>();
            sortedStatuByObject = new Dictionary<int, bool>();
            BufferSize = bufferSize;
        }

        // Get the topK neighbors of an object
        public List<KeyValuePair<int, double>> this[int indexOfObject]
        {
            get 
            { 
                // Make sure it is sorted before return
                if(sortedStatuByObject[indexOfObject]==false)
                {
                    SortAndRemoveNeighbors(indexOfObject);
                }
                return neighborsByObject[indexOfObject]; 
            }
        }

        public void AddSimilarityData(int indexOfObject, int indexOfNeighbor, double similarity)
        {
            if(!neighborsByObject.ContainsKey(indexOfObject))
            {
                neighborsByObject[indexOfObject] = new List<KeyValuePair<int, double>>();
            }

            neighborsByObject[indexOfObject].Add(new KeyValuePair<int,double>(indexOfNeighbor,similarity));
            sortedStatuByObject[indexOfObject] = false;

            if(neighborsByObject[indexOfObject].Count > BufferSize)
            {
                SortAndRemoveNeighbors(indexOfObject);
            }
        }
    }
}
