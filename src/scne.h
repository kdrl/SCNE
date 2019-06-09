#ifndef SCNE_H
#define SCNE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <locale>
#include <codecvt>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>

#include "cheaprand.h"

#define SIZE_TABLE_UNIGRAM 2000000
#define SIZE_CHUNK_PROGRESSBAR 1000

class SCNE {
private:
  const std::wstring corpus;
  const std::vector<std::wstring> vocabulary;
  const std::vector<int64_t> count_vocabulary;
  const int64_t n_max;
  const int64_t embed_dim;
  const int64_t random_seed;
  const int64_t epoch_num;
  const int64_t neg_num;
  const int64_t thread_num;
  const double learning_rate;
  const double sample_rate;
  const double power_freq;

  int64_t size_vocabulary;
  int64_t sum_count_vocabulary;

  int64_t n_min;

  std::unordered_map<std::wstring, int64_t> vocabulary2id;

  CheapRand cheaprand;
  int64_t* table_unigram;
  double* embeddings_targets;
  double* embeddings_contexts_left;
  double* embeddings_contexts_right;

public:
  SCNE(const std::wstring& _corpus,
       const std::vector<std::wstring>& _vocabulary,
       const std::vector<int64_t>& _count_vocabulary,
       const int64_t _n_max,
       const int64_t _embed_dim,
       const int64_t _random_seed,
       const int64_t _epoch_num,
       const int64_t _neg_num,
       const int64_t _thread_num,
       const double _learning_rate,
       const double _sample_rate,
       const double _power_freq);
  ~SCNE();
  void train(const std::string output_base_path);
  void save_vector(const std::string output_path);

private:
  void train_model_eachthread(const int64_t id_thread,
                              const int64_t i_wstr_start,
                              const int64_t length_str,
                              const int64_t thread_num);
  void initialize_parameters();
  void construct_unigramtable(const double power_freq);
};
#endif
