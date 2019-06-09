#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <chrono>

#include "cxxopts.hpp"
#include "lossycounting.h"
#include "scne.h"

int main(int argc, char* argv[]) {
  std::ios_base::sync_with_stdio(false);
  std::locale default_loc("en_US.UTF-8");
  std::locale::global(default_loc);
  std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype);
  std::wcout.imbue(ctype_default);
  std::wcin.imbue(ctype_default);

  std::string corpus_path, output_path;
  int64_t embed_dim, random_seed, epoch_num, neg_num, thread_num, voca_size, n_max, n_min;
  double learning_rate, sample_rate, power_freq, support_threshold, epsilon;

  cxxopts::Options options("SCNE", "SCNE");
  options.add_options()
    ("c,corpus_path", "Path of the training corpus", cxxopts::value(corpus_path))
    ("o,output_path", "Path of the output file, which will contain compositional n-grams with its embeddings", cxxopts::value(output_path))
    ("d,embed_dim", "Num of dim", cxxopts::value(embed_dim))
    ("r,random_seed", "Random seed", cxxopts::value(random_seed))
    ("e,epoch_num", "Number of epochs", cxxopts::value(epoch_num))
    ("n,neg_num", "Number of negative samples", cxxopts::value(neg_num))
    ("t,thread_num", "Number of threads", cxxopts::value(thread_num))
    ("m,voca_size", "Size of compositional n-gram set", cxxopts::value(voca_size))
    ("l,learning_rate", "Initial learning rate", cxxopts::value(learning_rate))
    ("s,sample_rate", "Sampling rate", cxxopts::value(sample_rate))
    ("p,power_freq", "Smoothing", cxxopts::value(power_freq))
    ("u,support_threshold", "Support threshold in lossy counting", cxxopts::value(support_threshold))
    ("i,epsilon", "Epsilon in lossy counting", cxxopts::value(epsilon))
    ("x,n_max", "Maximun size of n-gram to consider", cxxopts::value(n_max))
    ("a,n_min", "Minimun size of n-gram to consider", cxxopts::value(n_min))
    ;
  options.parse(argc, argv);

  std::wifstream fin_corpus(corpus_path);
  if (!fin_corpus.is_open()) {
    std::cout << "Invalid file name." << std::endl;
    return 0;
  }
  std::wstringstream wss;
  wss << fin_corpus.rdbuf();
  std::wstring corpus = wss.str();
  fin_corpus.close();

  std::vector<std::wstring> vocabulary;
  std::vector<int64_t> count_vocabulary;
  LossyCountingNgram counter(corpus, n_max, n_min, support_threshold, epsilon, thread_num);
  counter.count_ngram();
  counter.extract_top_ngram(vocabulary, count_vocabulary, voca_size);

  SCNE model(corpus, vocabulary, count_vocabulary, n_max,
             embed_dim, random_seed,
             epoch_num, neg_num, thread_num,
             learning_rate, sample_rate, power_freq);
  auto t1 = std::chrono::high_resolution_clock::now();
  model.train(output_path);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Training took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
            << " milliseconds.\n";
  model.save_vector(output_path);

  return 0;
}
