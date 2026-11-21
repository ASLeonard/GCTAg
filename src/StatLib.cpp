/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   Some statistic functions

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/
#include "StatLib.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/beta.hpp>
#include "cpu.h"

namespace StatLib{
    boost::math::chi_squared dist1(1);
    boost::math::chi_squared dist2(2); 
    boost::math::normal_distribution<> norm_dist(0.0, 1.0);

    double pchisqd1(double x){
        if(x > 0 && std::isfinite(x)){
            return std::erfc(std::sqrt(0.5 * x));
        }else{
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double pchisqd2(double x){
        if(x > 0 && std::isfinite(x)){
            return boost::math::cdf(boost::math::complement(dist2, x));
        }else{
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double qchisqd1(double x){
        if(x > 0 && x <= 1){
            return boost::math::quantile(boost::math::complement(dist1, x));
        }else if(x == 0){
            return std::numeric_limits<double>::infinity();
        }else{
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    Eigen::VectorXd weightBetaMAF(const Eigen::VectorXd& MAF, double weight_alpha, double weight_beta){
        boost::math::beta_distribution<> beta_weight(weight_alpha, weight_beta);
        int p_len = MAF.size();
        Eigen::VectorXd weights(p_len);
        for(int i = 0; i < p_len; i++){
            double af = MAF[i];
            weights[i] = boost::math::pdf(beta_weight, af);
        }
        return weights;
    }


    double pnorm(double x, bool bLowerTail){
        if(!std::isfinite(x)){
            return std::numeric_limits<double>::quiet_NaN();
        }
        if(bLowerTail){
            return boost::math::cdf(norm_dist, x);
        }else{
            return boost::math::cdf(boost::math::complement(norm_dist, x));
        }
    }
    /*
    //n rank size,  Z shall be n * n double memory.
    // return true, success; false, unsuccess
    bool rankContrast(int n, double * Z){
        double mean = 1.0 - (1.0 + n) / 2.0;

        //outer op
        std::vector<double> X(static_cast<size_t>(n) * n);
        for(int i = 0; i < n; i++){
            int base_index = i * n;
            for(int j = 0; j < n; j++){
                X[base_index + j] = pow(j + mean, i);
            }
        }

        std::vector<double> tau(n);
        gcta_blas_int n_lapack = (gcta_blas_int)n;
        gcta_blas_int info = 0;
        gcta_blas_int lda = n_lapack;
        gcta_blas_int lwork = -1;
        double work_query = 0;
        dgeqrf(&n_lapack, &n_lapack, X.data(), &lda, tau.data(), &work_query, &lwork, &info);
        lwork = (gcta_blas_int)work_query;
        std::vector<double> work(lwork);
        dgeqrf(&n_lapack, &n_lapack, X.data(), &lda, tau.data(), work.data(), &lwork, &info);
        double *c = Z;
        char side = 'L', trans = 'T';
        dormqr(&side, &trans, &n_lapack, &n_lapack, &n_lapack, X.data(), &lda, tau.data(), c, &lda, work.data(), &lwork, &info);
        if(info != 0){
            return false;
        }

        for(int i = 0; i < n; i++){
            double rdiv = 0;
            int base_index = i * n;
            for(int j = 0; j < n; j++){
                rdiv += c[base_index +j] * c[base_index + j];
            }
            rdiv = 1.0 / sqrt(rdiv);
            for(int j = 0; j < n; j++){
                c[base_index + j] = c[base_index + j] * rdiv;
            }
        }

        return true;
    }
    */

    bool rankContrast(int n, Eigen::MatrixXd &Z) {
        double mean = 1.0 - (1.0 + n) / 2.0;

        Eigen::MatrixXd X(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                X(j, i) = std::pow(j + mean, i);

        Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);
        Z = qr.householderQ().transpose() * Eigen::MatrixXd::Identity(n, n);

        for (int i = 0; i < n; i++)
            Z.row(i).normalize();

        return true;
    }


}
