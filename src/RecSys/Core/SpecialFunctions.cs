using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecSys.Core
{
    /// <summary>
    /// This class implements some special functions.
    /// </summary>
    public class SpecialFunctions
    {
        /// <summary>
        /// The inverse-logit function, i.e. Logistic function
        /// See http://en.wikipedia.org/wiki/Logistic_function
        /// and http://en.wikipedia.org/wiki/Logit
        /// </summary>
        /// <param name="alpha"></param>
        /// <returns>The inverse-logit output [0,1] of the intput value.</returns>
        public static double InverseLogit(double alpha)
        {
            double expOfAlpha = Math.Exp(alpha);
            return expOfAlpha / (1 + expOfAlpha);
        }
    }
}
