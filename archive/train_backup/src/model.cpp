//
// Created by Sanqiang Zhao on 10/10/16.
//

#include "model.h"
#include "util.h"
#include <memory>
#include <algorithm>
#include <fstream>
#include "data.h"
#include <unordered_set>

namespace entity2vec {

    model::model(std::shared_ptr<matrix> wi, std::shared_ptr<matrix> wo, std::shared_ptr<args> args, std::shared_ptr<data> data, uint32_t seed):
            neu1e(args->dim), rng(seed) {
        wi_ = wi;
        wo_ = wo;
        args_ = args;
        isz_ = wi->m_;
        osz_ = wo->m_;
        hsz_ = args->dim;
        negpos_word = 0;
        negpos_prod = 0;
        negpos_tag = 0;
        loss_ = 0.0;
        nexamples_ = 1;
        data_ = data;
        n_words_ = data_->word_size_;
        n_prods_ = data_->prod_size_;
        n_tags_ = data_->tag_size_;
    }

    real model::binaryLogistic(int64_t input, int64_t target, bool label, real lr) {
        real f = 0, g = 0, score = 0;
        for (int64_t i = 0; i < args_->dim; ++i) {
            f += wi_->getValue(input, i) * wo_->getValue(target, i);
        }
        if(f > MAX_SIGMOID){
            g = (real(label) - 1) * lr;
        } else if(f < -MAX_SIGMOID){
            g = (real(label) - 0) * lr;
        } else{
            score =  util::exp(f);
            g = (real(label) - score) * lr;
        }

        for (int64_t i = 0; i < args_->dim; ++i) {
            neu1e.incrementData(g * wo_->getValue(target, i), i);
        }

        if (label) {
            return -util::log(score);
        } else {
            return -util::log(1.0 - score);
        }
    }

    real model::negativeSampling(int64_t input, int64_t target, real lr) {
        real loss = 0.0;
        for (uint32_t n = 0; n <= args_->neg; n++) {
            if (n == 0) {
                loss += binaryLogistic(input, target, true, lr);
            } else {
                int64_t neg_target = getNegative(input, target);
                if (neg_target == -1)
                    return 0;
                loss += binaryLogistic(input, neg_target, false, lr);
            }
        }
        return loss;
    }

    int64_t model::getNegative(int64_t input, int64_t target) {
        int64_t negative = -1;
        int64_t cnt = 0;
        if(checkIndexType(input) == 0 && checkIndexType(target) == 0){ //word-word is word
            do {
                negative = word_negatives[negpos_word % word_negatives.size()];
                negpos_word = (negpos_word + 1) % word_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
            } while (target == negative);
        }else if(checkIndexType(input) == 0 && checkIndexType(target) == 1){

            do {
                negative = prod_negatives[negpos_prod % prod_negatives.size()];
                negpos_prod = (negpos_prod + 1) % prod_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
                if(!data_->checkCorPair(input, negative, 1)){
                    break;
                }
            } while (1);
            //negative += data_->word_size_;
            negative = transform_dic2matrix(negative, 1);
        }else if(checkIndexType(input) == 1 && checkIndexType(target) == 0){
            do {
                negative = word_negatives[negpos_word % word_negatives.size()];
                negpos_word = (negpos_word + 1) % word_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
                if(!data_->checkCorPair(negative, input - data_->word_size_, 1)){
                    break;
                }
            } while (1);
        }else if(checkIndexType(input) == 0 && checkIndexType(target) == 2){
            do {
                negative = tag_negatives[negpos_tag % tag_negatives.size()];
                negpos_tag = (negpos_tag + 1) % tag_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
                if(!data_->checkCorPair(input, negative, 2)){
                    break;
                }
            } while (1);
            //negative += data_->word_size_ + data_->prod_size_;
            negative = transform_dic2matrix(negative, 2);
        }else if(checkIndexType(input) == 2 && checkIndexType(target) == 0){
            do {
                negative = word_negatives[negpos_word % word_negatives.size()];
                negpos_word = (negpos_word + 1) % word_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
                if(!data_->checkCorPair(negative, input - data_->word_size_ - data_->prod_size_, 2)){
                    break;
                }
            } while (1);
        }else if(checkIndexType(input) == 1 && checkIndexType(target) == 2){
            do {
                negative = tag_negatives[negpos_tag % tag_negatives.size()];
                negpos_tag = (negpos_tag + 1) % tag_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
                if(!data_->checkCorPair(negative, input- data_->word_size_, 3)){
                    break;
                }
            } while (1);
            //negative += data_->word_size_ + data_->prod_size_;
            negative = transform_dic2matrix(negative, 2);
        }else if(checkIndexType(input) == 2 && checkIndexType(target) == 1){
            do {
                negative = prod_negatives[negpos_prod % prod_negatives.size()];
                negpos_prod = (negpos_prod + 1) % prod_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
                if(!data_->checkCorPair(input  - data_->word_size_ - data_->prod_size_, negative, 3)){
                    break;
                }
            } while (1);
            //negative += data_->word_size_;
            negative = transform_dic2matrix(negative, 1);
        }
        return negative;
    }

    void model::initTableNegatives() {
        const std::vector<uint32_t> counts = data_->getWordCounts();
        real z = 0.0;
        for (size_t i = 0; i < counts.size(); i++) {
            z += pow(counts[i], 0.75);
        }
        for (size_t i = 0; i < counts.size(); i++) {
            real c = pow(counts[i], 0.75);
            for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                word_negatives.push_back(i);
            }
        }
        std::shuffle(word_negatives.begin(), word_negatives.end(), rng);

        if(args_->prod_flag){
            const std::vector<uint32_t> counts_prod = data_->getProdCounts();
            real z_prod = 0.0;
            for (size_t i = 0; i < counts_prod.size(); i++) {
                z_prod += pow(counts_prod[i], 0.75);
            }
            for (size_t i = 0; i < counts_prod.size(); i++) {
                real c = pow(counts_prod[i], 0.75);
                for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                    prod_negatives.push_back(i);
                }
            }
            std::shuffle(prod_negatives.begin(), prod_negatives.end(), rng);

            const std::vector<uint32_t> counts_tag = data_->getTagCounts();
            real z_tag = 0.0;
            for (size_t i = 0; i < counts_tag.size(); i++) {
                z_tag += pow(counts_tag[i], 0.75);
            }
            for (size_t i = 0; i < counts_tag.size(); i++) {
                real c = pow(counts_tag[i], 0.75);
                for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                    tag_negatives.push_back(i);
                }
            }
            std::shuffle(tag_negatives.begin(), tag_negatives.end(), rng);
        }
    }

    void model::initWordNegSampling() {
        initTableNegatives();
    }

    void model::update(int64_t input, int64_t target, real lr) {
        neu1e.zero();
        loss_ += negativeSampling(input, target, lr);

        nexamples_ += 1;
        wi_->addRow(neu1e, input, 1.0);

    }

    real model::getLoss() const {
        return loss_ / nexamples_;
    }

    void model::load(std::istream &in) {
        for (int32_t i = 0; i < NEGATIVE_TABLE_SIZE; ++i) {
            int32_t temp;
            in.read((char*) &temp, sizeof(int32_t));
            word_negatives.push_back(temp);
        }
    }

    void model::save(std::ostream &out) {
        for (int32_t i = 0; i < NEGATIVE_TABLE_SIZE; ++i) {
            out.write((char*) &(word_negatives[i]), sizeof(int32_t));
        }
    }

    uint8_t model::checkIndexType(int64_t index) {
        if (index < n_words_){
            return 0;
        }else if(index < n_words_+n_prods_){
            return 1;
        }else if(index < n_words_+n_prods_+n_tags_){
            return 2;
        }
    }

    int64_t model::transform_matrix2dic(int64_t index) {
        if(checkIndexType(index) == 0){
            return index;
        }else if(checkIndexType(index) == 1){
            return index - n_words_;
        }else if(checkIndexType(index) == 2){
            return index - n_words_ - n_prods_;
        }
    }

    int64_t model::transform_dic2matrix(int64_t index, uint8_t mode) {
        if(mode == 0){
            return index;
        }else if(mode == 1){
            return index + n_words_;
        }else if(mode == 2){
            return index + n_words_ + n_prods_;
        }
    }

}