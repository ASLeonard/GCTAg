/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   New implementation: covar information

   Developed by Zhili Zheng<zhilizheng@outlook.com>, 2017

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/

#include "Covar.h"
#include "Logger.h"
#include "OptionIO.h"
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <ranges>
#include <numeric>
#include "utils.hpp"
#include "StatLib.h"

using std::to_string;

bool common_3vector(vector<string> **idListP, vector<vector<double>> **valueP,
        vector<string> *target_id, vector<vector<double>> **targetP){

    // can't be (*idlist)[0], this get the first element 
    vector<string> &s1 = *(idListP[0]);
    vector<string> &s2 = *(idListP[1]);
    vector<string> &s3 = *(idListP[2]);
    vector<string> &s4 = *(idListP[3]); 
    if(s1.size() == 0 || s2.size() == 0){
        return false;
    }

    vector<int> s1_index, s2_index;
    vector_commonIndex_sorted1(s1, s2, s1_index, s2_index);
    int remain_size = s1_index.size();
    if(remain_size == 0){
        return false;
    }
    vector<string> sample_id;
    sample_id.reserve(remain_size);
    std::ranges::transform(s1_index, std::back_inserter(sample_id), [&s1](int pos){return s1[pos];}); 

    vector<int> s3_index, s4_index;
    if(s3.size() != 0){
        vector<int> sa_index;
        vector_commonIndex_sorted1(sample_id, s3, sa_index, s3_index);
        remain_size = sa_index.size();
        if(remain_size == 0){
            return false;
        }
        vector<int> temp_s1, temp_s2;
        temp_s1.reserve(remain_size);
        temp_s2.reserve(remain_size);
        std::ranges::transform(sa_index, std::back_inserter(temp_s1), [&s1_index](int pos){return s1_index[pos];});
        s1_index = std::move(temp_s1);

        std::ranges::transform(sa_index, std::back_inserter(temp_s2), [&s2_index](int pos){return s2_index[pos];});
        s2_index = std::move(temp_s2);

        sample_id.clear();
        sample_id.reserve(remain_size);
        std::ranges::transform(s1_index, std::back_inserter(sample_id), [&s1](int pos){return s1[pos];}); 
        
        if(s4.size() != 0){
            vector<int> sc_index;
            vector_commonIndex_sorted1(sample_id, s4, sc_index, s4_index);
            remain_size = sc_index.size();
            if(remain_size == 0){
              return false;
            }
            vector<int> temp_s3;
            temp_s1.clear();
            temp_s2.clear();
            temp_s1.reserve(remain_size);
            temp_s2.reserve(remain_size);
            temp_s3.reserve(remain_size);
            
            std::ranges::transform(sc_index, std::back_inserter(temp_s1), [&s1_index](int pos){return s1_index[pos];});
            s1_index = std::move(temp_s1);
            
            std::ranges::transform(sc_index, std::back_inserter(temp_s2), [&s2_index](int pos){return s2_index[pos];});
            s2_index = std::move(temp_s2);
            
            std::ranges::transform(sc_index, std::back_inserter(temp_s3), [&s3_index](int pos){return s3_index[pos];});
            s3_index = std::move(temp_s3);
            
        }
    }

    target_id->resize(remain_size);
    std::ranges::transform(s1_index, target_id->begin(), [&s1](int pos){return s1[pos];});

    auto &value1 = *valueP[0];
    auto &value2 = *valueP[1];
    auto &value3 = *valueP[2];
    auto &value4 = *valueP[3];
    auto &targetvalue1 = *targetP[0];
    auto &targetvalue2 = *targetP[1];
    auto &targetvalue3 = *targetP[2];
    auto &targetvalue4 = *targetP[3];

    vector<vector<double>> *values[] = {&value1, &value2, &value3, &value4};
    vector<int> *indicies[] = {&s1_index, &s2_index, &s3_index, &s4_index};
    vector<vector<double>> *targetvalues[] = {&targetvalue1, &targetvalue2, &targetvalue3, &targetvalue4};

    for(int i = 0; i < 4; i++){
        auto &temp_value = *(values[i]);
        auto &temp_index = *(indicies[i]);
        auto &temp_target = *(targetvalues[i]);
        for(int j = 0; j < temp_value.size(); j++){
            auto &pvalue = temp_value[j];
            vector<double> t_value(temp_index.size());
            std::ranges::transform(temp_index, t_value.begin(), [&pvalue](int pos){return pvalue[pos];});
            temp_target[j] = std::move(t_value);
        }
    }

    return true;
}



Covar::Covar(const gcta_cli::PhenoCovarOptions& opts) {
    vector<string> samples_qcovar, samples_covar, samples_rcovar, samples_envir;
    vector<vector<double>> qcovar, covar, rcovar, envir;

    if (!opts.qcovar.empty()) {
        requireFileReadable("--qcovar", opts.qcovar);
        LOGGER.i(0, "Reading quantitative covariates from [" + opts.qcovar + "].");
        read_covar(opts.qcovar, samples_qcovar, &qcovar, nullptr, nullptr, opts.covar_maxlevel);
        if (hasVectorDuplicate(samples_qcovar)) {
            LOGGER.e(0, "duplicated FID+IID in the covariate file.");
        }
        LOGGER.i(0, to_string(qcovar.size()) + " covariates of " + to_string(samples_qcovar.size()) + " samples to be included.");
    }

    if (!opts.covar.empty()) {
        requireFileReadable("--covar", opts.covar);
        LOGGER.i(0, "Reading discrete covariates from [" + opts.covar + "].");
        read_covar(opts.covar, samples_covar, &covar, &labels_covar, nullptr, opts.covar_maxlevel);
        if (hasVectorDuplicate(samples_covar)) {
            LOGGER.e(0, "duplicated FID+IID in the covariate file.");
        }
        LOGGER.i(0, to_string(covar.size()) + " covariates of " + to_string(samples_covar.size()) + " samples to be included.");
    }

    this->rcovar.resize(rcovar.size());
    this->covar.resize(covar.size());
    this->qcovar.resize(qcovar.size());
    this->envir.resize(envir.size());  

    int flags_samples = (samples_qcovar.size() != 0) + (samples_covar.size() != 0) * 10 + 
        (samples_rcovar.size() != 0) * 100 + (samples_envir.size() != 0) * 1000;


    vector<string> *idList[4];
    vector<vector<double>> *values[4];
    vector<vector<double>> *targets[4];
    
    switch(flags_samples){
        case 0:
            return;
        case 1:
            sample_id = std::move(samples_qcovar);
            this->qcovar = std::move(qcovar);
            return;
        case 10:
            sample_id = std::move(samples_covar);
            this->covar = std::move(covar);
            setCovarMapping(false);
            return;
        case 100:
            sample_id = std::move(samples_rcovar);
            this->rcovar = std::move(rcovar);
            setCovarMapping(true);
            return;
        case 1000:
            sample_id = std::move(samples_envir);
            this->envir = std::move(envir);
            return;
        case 11:
            idList[0] = &samples_qcovar;
            idList[1] = &samples_covar;
            idList[2] = &samples_rcovar;
            idList[3] = &samples_envir;
            values[0] = &qcovar;
            values[1] = &covar;
            values[2] = &rcovar;
            values[3] = &envir;
            targets[0] = &(this->qcovar);
            targets[1] = &(this->covar);
            targets[2] = &(this->rcovar);
            targets[3] = &(this->envir);
            break;
        case 101:
            idList[0] = &samples_qcovar;
            idList[2] = &samples_covar;
            idList[1] = &samples_rcovar;
            idList[3] = &samples_envir;
            values[0] = &qcovar;
            values[2] = &covar;
            values[1] = &rcovar;
            values[3] = &envir;
            targets[0] = &(this->qcovar);
            targets[2] = &(this->covar);
            targets[1] = &(this->rcovar);
            targets[3] = &(this->envir);
            break;
        case 110:
            idList[2] = &samples_qcovar;
            idList[1] = &samples_covar;
            idList[0] = &samples_rcovar;
            idList[3] = &samples_envir;
            values[2] = &qcovar;
            values[1] = &covar;
            values[0] = &rcovar;
            values[3] = &envir;
            targets[2] = &(this->qcovar);
            targets[1] = &(this->covar);
            targets[0] = &(this->rcovar);
            targets[3] = &(this->envir);
            break;
        case 1001:  
            idList[0] = &samples_qcovar;
            idList[2] = &samples_covar;
            idList[3] = &samples_rcovar;
            idList[1] = &samples_envir;
            values[0] = &qcovar;
            values[2] = &covar;
            values[3] = &rcovar;
            values[1] = &envir;
            targets[0] = &(this->qcovar);
            targets[2] = &(this->covar);
            targets[3] = &(this->rcovar);
            targets[1] = &(this->envir);
            break;
        case 1010:
            idList[2] = &samples_qcovar;
            idList[0] = &samples_covar;
            idList[3] = &samples_rcovar;
            idList[1] = &samples_envir;
            values[2] = &qcovar;
            values[0] = &covar;
            values[3] = &rcovar;
            values[1] = &envir;
            targets[2] = &(this->qcovar);
            targets[0] = &(this->covar);
            targets[3] = &(this->rcovar);
            targets[1] = &(this->envir);
            break;
        case 1100:
            idList[2] = &samples_qcovar;
            idList[3] = &samples_covar;
            idList[0] = &samples_rcovar;
            idList[1] = &samples_envir;
            values[2] = &qcovar;
            values[3] = &covar;
            values[0] = &rcovar;
            values[1] = &envir;
            targets[2] = &(this->qcovar);
            targets[3] = &(this->covar);
            targets[0] = &(this->rcovar);
            targets[1] = &(this->envir);
            break;
        case 111:
            idList[0] = &samples_qcovar;
            idList[1] = &samples_covar;
            idList[2] = &samples_rcovar;
            idList[3] = &samples_envir;
            values[0] = &qcovar;
            values[1] = &covar;
            values[2] = &rcovar;
            values[3] = &envir;
            targets[0] = &(this->qcovar);
            targets[1] = &(this->covar);
            targets[2] = &(this->rcovar);
            targets[3] = &(this->envir);
            break;
        case 1011:
            idList[0] = &samples_qcovar;
            idList[1] = &samples_covar;
            idList[3] = &samples_rcovar;
            idList[2] = &samples_envir;
            values[0] = &qcovar;
            values[1] = &covar;
            values[3] = &rcovar;
            values[2] = &envir;
            targets[0] = &(this->qcovar);
            targets[1] = &(this->covar);
            targets[3] = &(this->rcovar);
            targets[2] = &(this->envir);
            break;
        case 1101:
            idList[0] = &samples_qcovar;
            idList[3] = &samples_covar;
            idList[1] = &samples_rcovar;
            idList[2] = &samples_envir;
            values[0] = &qcovar;
            values[3] = &covar;
            values[1] = &rcovar;
            values[2] = &envir;
            targets[0] = &(this->qcovar);
            targets[3] = &(this->covar);
            targets[1] = &(this->rcovar);
            targets[2] = &(this->envir);
            break;
        case 1110:
            idList[3] = &samples_qcovar;
            idList[1] = &samples_covar;
            idList[0] = &samples_rcovar;
            idList[2] = &samples_envir;
            values[3] = &qcovar;
            values[1] = &covar;
            values[0] = &rcovar;
            values[2] = &envir;
            targets[3] = &(this->qcovar);
            targets[1] = &(this->covar);
            targets[0] = &(this->rcovar);
            targets[2] = &(this->envir);
            break;
        case 1111:
            idList[0] = &samples_qcovar;
            idList[1] = &samples_covar;
            idList[2] = &samples_rcovar;
            idList[3] = &samples_envir;
            values[0] = &qcovar;
            values[1] = &covar;
            values[2] = &rcovar;
            values[3] = &envir;
            targets[0] = &(this->qcovar);
            targets[1] = &(this->covar);
            targets[2] = &(this->rcovar);
            targets[3] = &(this->envir);
            break;
        default:
            LOGGER.e(0, "unexpected error in the covariate file. Please check.");
    }
    if(common_3vector(idList, values, &sample_id, targets)){
        LOGGER.i(0, to_string(qcovar.size()) + " qcovar, " + to_string(covar.size()) + " covar and " + to_string(rcovar.size()) + " rcovar to be included.");
        setCovarMapping(false);
        setCovarMapping(true);
    }else{
        LOGGER.e(0, "no covariate to be included.");
    }
    LOGGER.i(0, to_string(sample_id.size()) + " common individuals among the covariate files to be included.");
}

const vector<string>& Covar::getSampleID() const{
    return this->sample_id;
}

bool Covar::setCovarMapping(bool is_rcovar, vector<vector<string>> *map_order){
    string err_string;
    if(is_rcovar){
        err_string = "can't specify inconsistent or single ranked covariate labels.";
    }else{
        err_string = "can't specify inconsistent or single covariate labels.";
    }
    if(sample_id.size() == 0 || covar.size() == 0){
        return false;
    }
    if(map_order){
        if(map_order->size() != covar.size()){
            LOGGER.e(0, err_string);
            return false;
        }
    }

    int expand_col_covar = 0;
    vector<int> start_col_X;
    start_col_X.push_back(0);

    // TODO: labels_covar_mapping and labels_rcovar_mapping are effectively
    // fixed-size 2D tables (all inner vectors have the same length n).
    // Replace with Eigen::MatrixXd to eliminate the jagged allocation and
    // simplify the index arithmetic in getCovarX / getCovarXRaw.
    auto *plabels_covar = &this->labels_covar;
    auto *pcovar = &this->covar;
    auto *plabels_covar_mapping = &this->labels_covar_mapping;
    if(is_rcovar){
        plabels_covar = &this->labels_rcovar;
        pcovar = &this->rcovar;
        plabels_covar_mapping = &this->labels_rcovar_mapping;
    }

    auto &labels_covar = *plabels_covar;
    auto &covar = *pcovar;
    auto &labels_covar_mapping = *plabels_covar_mapping;

    for(auto & label : labels_covar){
        expand_col_covar += label["LABEL_MAX_VALUE"];
        start_col_X.push_back(expand_col_covar);
    }

    labels_covar_mapping.resize(expand_col_covar);

    for(int i = 0; i < covar.size(); i++){
        auto &map_item = labels_covar[i];
        vector<string> elements;
        for(auto const& t_map : map_item){
            const string &key = t_map.first;
            if(key != "LABEL_MAX_VALUE"){
                elements.push_back(key);
            }
        }
        std::ranges::sort(elements);
        if(elements.size() == 1){
            LOGGER.e(0, "column " + to_string(i) + ", " + err_string);
            return false;
        }
        if(map_order){
            vector<string> &pelements = (*map_order)[i];
            vector<string> mapper;
            for(auto const &element : pelements){
                if(std::binary_search(elements.begin(), elements.end(), element)){
                    mapper.push_back(element);
                }
            }
            if(elements.size() != mapper.size()){
                LOGGER.e(0, "column " + to_string(i) + ", " + err_string);
                return false;
            }
            elements = mapper;
        }

        if(!is_rcovar){
            for(int j = 1; j < elements.size(); j++){
                vector<double> item(elements.size(), 0);
                item[map_item[elements[j]]] = 1.0;
                labels_covar_mapping[start_col_X[i] + j - 1] = item;
            }
        }else{
            int n = elements.size();
            Eigen::MatrixXd Z(n, n);
            if(StatLib::rankContrast(n, Z)){
                for(int j = 1; j < n; j++){
                    std::vector<double> item(n);
                    for(auto const& [key, idx] : map_item){
                        if(key != "LABEL_MAX_VALUE")
                            item[idx] = Z(idx, j);  // direct indexing, no manual offset arithmetic
                    }
                    labels_covar_mapping[start_col_X[i] + j - 1] = item;
                }
            }else{
                return false;
            }
        }
    }


    return true;
}

bool Covar::getCommonSampleIndex(const vector<string> &sampleIDs, vector<uint32_t> &keep_index, vector<uint32_t> &covar_index){
    if(sampleIDs.size() == 0 || (!hasCovar())){
        return false;
    }
    vector_commonIndex_sorted1(sampleIDs, sample_id, keep_index, covar_index);
    if(keep_index.size() == 0){
        return false;
    }
    return true;
}

bool Covar::hasCovar(){
    int total_col_covar = qcovar.size() + covar.size() + rcovar.size() + envir.size();
    if(sample_id.size() == 0 || total_col_covar == 0){
        return false;
    }else{
        return true;
    }
}

bool Covar::hasEnvir(){
    if (sample_id.size() == 0 || envir.size() == 0){
        return false;
    }else{
        return true;
    }
}

bool Covar::getCovarX(const vector<string> &sampleIDs, vector<double> &X, vector<uint32_t> &keep_index){
    vector<uint32_t> covar_index;
    if(!getCommonSampleIndex(sampleIDs, keep_index, covar_index)){
        return false;
    }

    int common_sample_size = covar_index.size();

    vector<int> start_col_X;
    start_col_X.push_back(0);
    int expand_col_covar = 0;
    
    for(int i = 0; i < qcovar.size(); i++){
        expand_col_covar++;
        start_col_X.push_back(expand_col_covar);
    }

    for(auto & label : labels_covar){
        expand_col_covar += label["LABEL_MAX_VALUE"];
        start_col_X.push_back(expand_col_covar);
    }

    for(auto & label : labels_rcovar){
        expand_col_covar += label["LABEL_MAX_VALUE"];
        start_col_X.push_back(expand_col_covar);
    }

    for (int i = 0; i < envir.size(); i++){ 
        expand_col_covar++;
        start_col_X.push_back(expand_col_covar);
    }

    X.resize(expand_col_covar * common_sample_size);

    for(int i = 0; i < qcovar.size(); i++){
        auto &covarp = this->qcovar[i];
        int base_pos = i * common_sample_size;
        std::transform(covar_index.begin(), covar_index.end(), X.begin() + base_pos,
                [&covarp](int pos){return covarp[pos];});
    }

    int base_covar = qcovar.size();
    int map_index = 0;
    for(int i = 0; i < covar.size(); i++){
        auto &covarp = this->covar[i];
        auto &labelp = this->labels_covar[i];
        int base_pos = (start_col_X[base_covar + i]) * common_sample_size;
        for(int j = 0; j < labelp["LABEL_MAX_VALUE"]; j++){
            auto &cur_table = labels_covar_mapping[map_index];
            std::ranges::transform(covar_index, X.begin() + base_pos + j * common_sample_size, [&cur_table, &covarp](int pos){ return cur_table[(int)covarp[pos]];});
            map_index++;
        }
    }

    int base_rcovar = qcovar.size() + covar.size();
    int map_rindex = 0;
    for(int i = 0; i < rcovar.size(); i++){
        auto &covarp = this->rcovar[i];
        auto &labelp = this->labels_rcovar[i];
        int base_pos = (start_col_X[base_rcovar + i]) * common_sample_size;
        for(int j = 0; j < labelp["LABEL_MAX_VALUE"]; j++){
            auto &cur_table = labels_rcovar_mapping[map_rindex];
            std::ranges::transform(covar_index, X.begin() + base_pos + j * common_sample_size, [&cur_table, &covarp](int pos){ return cur_table[(int)covarp[pos]];});
            //std::transform(covarp.begin(), covarp.end(), X.begin() + base_pos + j * common_sample_size, [&cur_table](double covar){ return cur_table[(int)covar];});
            map_rindex++;
        }
    }

    int base_envir = qcovar.size() + covar.size() + rcovar.size();
    for(int i = 0; i < envir.size(); i++){
        auto &covarp = this->envir[i];
        int base_pos = (start_col_X[base_envir + i]) * common_sample_size;
        std::ranges::transform(covar_index, X.begin() + base_pos,
                       [&covarp](int pos){return covarp[pos];});
    }

    return true;
}

bool Covar::getCovarXRaw(const vector<string> &sampleIDs, vector<double> &X, vector<uint32_t> &keep_index){
    vector<uint32_t> covar_index;
    if(!getCommonSampleIndex(sampleIDs, keep_index, covar_index)){
        return false;
    }

    uint64_t common_sample_size = covar_index.size();
    uint64_t total_col_covar = qcovar.size() + covar.size() + rcovar.size();
    X.resize(total_col_covar * common_sample_size);

    vector<vector<double>> *covars[] = {&qcovar, &covar, &rcovar};
    vector<uint64_t> base_X_positions = {0, qcovar.size(), qcovar.size() + covar.size()};
    vector<uint64_t> look_X_positions = {qcovar.size(), qcovar.size() + covar.size(), total_col_covar};


    #pragma omp parallel for
    for(int i = 0; i < total_col_covar; i++){
        //get which covar
        auto const it = std::lower_bound(look_X_positions.begin(), look_X_positions.end(), i + 1);
        int j = std::distance(look_X_positions.begin(), it);

        auto &pcovar = *covars[j];
        int base_pos = i * common_sample_size;
        auto &t_covar = pcovar[i - base_X_positions[j]];
        std::transform(covar_index.begin(), covar_index.end(), X.begin() + base_pos, 
                [&t_covar](int pos){return t_covar[pos];});
    }

    return true;
/*  // can't be paralleled
    for(int j = 0; j < sizeof(covars) / sizeof(covars[0]); j++){
        auto &pcovar = *covars[j];
        int base_position = base_X_positions[j] * common_sample_size;
        #pragma omp parallel for
        for(int i = 0; i < pcovar.size(); i++){ 
            auto &t_covar = pcovar[i];
            int temp_begin_pos = base_position + i * common_sample_size;  
            std::transform(covar_index.begin(), covar_index.end(), X.begin() + temp_begin_pos, 
                    [&t_covar](int pos){return t_covar[pos];});
        }
    }
*/
}

bool Covar::getEnvirX(const vector<string> &sampleIDs, vector<double> &X, vector<uint32_t> &keep_index){
  vector<uint32_t> covar_index;
  if(!getCommonSampleIndex(sampleIDs, keep_index, covar_index)){
    return false;
  }
  
  int common_sample_size = covar_index.size();
  
  vector<int> start_col_X;
  start_col_X.push_back(0);
  int expand_col_covar = 0;
  
  for (int i = 0; i < envir.size(); i++){ 
    expand_col_covar++;
    start_col_X.push_back(expand_col_covar);
  }
  
  X.resize(expand_col_covar * common_sample_size);
  
  for(int i = 0; i < envir.size(); i++){
    auto &covarp = this->envir[i];
    int base_pos = i * common_sample_size;
    std::ranges::transform(covar_index, X.begin() + base_pos,
                   [&covarp](int pos){return covarp[pos];});
  }
  return true;
}

namespace {

inline bool is_missing_covar_token(std::string_view sv) noexcept {
    const size_t n = sv.size();
    if (n == 1) return sv[0] == '.';
    if (n == 2) {
        if (sv[0] == '-' && sv[1] == '9') return true;
        return gcta_text_io::ascii_toupper(sv[0]) == 'N' &&
               gcta_text_io::ascii_toupper(sv[1]) == 'A';
    }
    if (n == 3) {
        return gcta_text_io::ascii_toupper(sv[0]) == 'N' &&
               gcta_text_io::ascii_toupper(sv[1]) == 'A' &&
               gcta_text_io::ascii_toupper(sv[2]) == 'N';
    }
    return false;
}

inline bool is_covar_header_first_tok(std::string_view sv) noexcept {
    if (sv.empty()) return false;
    if (sv[0] == '#') return true;
    return sv.size() == 3 &&
           gcta_text_io::ascii_toupper(sv[0]) == 'F' &&
           gcta_text_io::ascii_toupper(sv[1]) == 'I' &&
           gcta_text_io::ascii_toupper(sv[2]) == 'D';
}

} // namespace

void Covar::read_covar(string filename, vector<string>& sub_list, vector<vector<double>>* covar, vector<map<string, int>>* labels, vector<int>* keep_row_p, int max_level){
    const string err_string = "[" + filename + "].";
    gcta_text_io::FastLineReader reader;
    if (!reader.open(filename)) {
        LOGGER.e(0, "can't read " + err_string);
    }

    int nkeep = 0;
    int last_keep = 0;
    if (keep_row_p) {
        nkeep = static_cast<int>(keep_row_p->size());
        last_keep = (*keep_row_p)[static_cast<size_t>(nkeep - 1)];
    }
    int least_col = 2;
    bool has_covar = false;
    if (covar) {
        least_col = 3;
        has_covar = true;
    }
    last_keep += 2;

    std::string_view line;
    if (!reader.next_line(line)) {
        LOGGER.e(0, "less than " + to_string(least_col) + " columns in " + err_string);
    }

    std::vector<std::string_view> toks;
    toks.reserve(64);
    gcta_text_io::split_ws_exact(line, toks);
    const int ncol = static_cast<int>(toks.size());

    if (ncol < least_col) {
        LOGGER.e(0, "less than " + to_string(least_col) + " columns in " + err_string);
    }
    if (last_keep > ncol) {
        LOGGER.e(0, "can't read " + to_string(last_keep) + "th column from " + err_string);
    }

    const bool has_header = is_covar_header_first_tok(toks[0]);
    int line_number = has_header ? 1 : 0;

    vector<int> keep_col;
    if (nkeep == 0) {
        keep_col.resize(static_cast<size_t>(ncol - 2));
        std::iota(keep_col.begin(), keep_col.end(), 2);
        nkeep = static_cast<int>(keep_col.size());
    } else {
        keep_col.resize(static_cast<size_t>(nkeep));
        std::ranges::transform(*keep_row_p, keep_col.begin(), [](int value) { return value + 1; });
    }

    if (has_covar) {
        covar->resize(static_cast<size_t>(nkeep));
    } else {
        nkeep = 0;
    }

    bool is_factor = false;
    if (labels) {
        is_factor = true;
        labels->resize(static_cast<size_t>(nkeep));
        for (int i = 0; i < nkeep; i++) {
            (*labels)[static_cast<size_t>(i)]["LABEL_MAX_VALUE"] = -1;
        }
    }

    const int n_item = max_level;
    vector<double> covar_temp(static_cast<size_t>(nkeep));
    std::string factor_buf;
    factor_buf.reserve(64);

    size_t sample_lines = 0;
    size_t sample_bytes = 0;
    bool reserved = false;

    auto process_row = [&](std::string_view cur_line) {
        line_number++;
        const int cur_ncol = static_cast<int>(toks.size());
        if (cur_ncol < last_keep) {
            LOGGER.e(0, "can't read " + to_string(last_keep) + "th column of line " + to_string(line_number) + " from " + err_string);
        } else if (cur_ncol != ncol) {
            LOGGER.w(0, "inconsistent column number in line " + to_string(line_number) + " from " + err_string);
        }

        if (!reserved && sample_lines < 16) {
            sample_bytes += cur_line.size() + 1;
            ++sample_lines;
            if (sample_lines == 16 && reader.file_size() > 0) {
                reserved = true;
                const size_t avg_line = std::max<size_t>(sample_bytes / sample_lines, 6);
                const size_t est_rows = static_cast<size_t>(reader.file_size() / avg_line) + 64;
                sub_list.reserve(sub_list.size() + est_rows);
                if (has_covar) {
                    for (int i = 0; i < nkeep; ++i) {
                        (*covar)[static_cast<size_t>(i)].reserve(
                            (*covar)[static_cast<size_t>(i)].size() + est_rows);
                    }
                }
            }
        }

        bool is_skip = false;
        if (has_covar) {
            for (int col_index = 0; col_index < nkeep; col_index++) {
                const std::string_view sv = toks[static_cast<size_t>(keep_col[static_cast<size_t>(col_index)])];
                if (is_missing_covar_token(sv)) {
                    is_skip = true;
                    break;
                }

                double temp_item = 0.0;
                if (is_factor) {
                    factor_buf.resize(sv.size());
                    for (size_t k = 0; k < sv.size(); ++k) {
                        factor_buf[k] = gcta_text_io::ascii_toupper(sv[k]);
                    }
                    auto& label_map = (*labels)[static_cast<size_t>(col_index)];
                    const auto it = label_map.find(factor_buf);
                    if (it != label_map.end()) {
                        temp_item = static_cast<double>(it->second);
                    } else {
                        int& max_val = label_map["LABEL_MAX_VALUE"];
                        max_val += 1;
                        if (max_val > n_item) {
                            LOGGER.e(0, "too many levels in covariate #" + to_string(col_index + 1) + ". You may fit it as a quantitative covariate using --qcovar.");
                        }
                        label_map.emplace(factor_buf, max_val);
                        temp_item = static_cast<double>(max_val);
                    }
                } else {
                    const char* first = sv.data();
                    const char* last  = first + sv.size();
                    if (first < last && *first == '+') ++first;
                    auto [ptr, ec] = std::from_chars(first, last, temp_item);
                    if (ec != std::errc{} || ptr != last || !std::isfinite(temp_item)) {
                        LOGGER.error(0, "line {} contains non-numeric values in {}", line_number, err_string);
                    }
                }
                covar_temp[static_cast<size_t>(col_index)] = temp_item;
            }
        }

        if (!is_skip) {
            std::string id;
            id.reserve(toks[0].size() + 1 + toks[1].size());
            id.append(toks[0]);
            id.push_back('\t');
            id.append(toks[1]);
            sub_list.push_back(std::move(id));
            for (int i = 0; i < nkeep; i++) {
                (*covar)[static_cast<size_t>(i)].push_back(covar_temp[static_cast<size_t>(i)]);
            }
        }
    };

    if (!has_header) {
        process_row(line);
    }
    while (reader.next_line(line)) {
        gcta_text_io::split_ws_exact(line, toks);
        process_row(line);
    }
}


//template void Covar::read_covar(string filename, vector<string>& sub_list, vector<vector<int8_t>>* covar, vector<map<string, int8_t>>* labels, vector<int>* keep_row_p);
//template void Covar::read_covar(string filename, vector<string>& sub_list, vector<vector<double>>* covar, vector<map<string, double>>* labels, vector<int>* keep_row_p);
