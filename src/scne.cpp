#include "scne.h"

SCNE::SCNE(const std::wstring& _corpus,
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
           const double _power_freq)
  : corpus(_corpus),
    vocabulary(_vocabulary),
    count_vocabulary(_count_vocabulary),
    n_max(_n_max),
    embed_dim(_embed_dim),
    random_seed(_random_seed),
    epoch_num(_epoch_num),
    neg_num(_neg_num),
    thread_num(_thread_num),
    learning_rate(_learning_rate),
    sample_rate(_sample_rate),
    power_freq(_power_freq)
{
  assert(n_max > 0);
  assert(embed_dim > 0);
  assert(random_seed > 0);
  assert(epoch_num >= 0);
  assert(neg_num >= 0);
  assert(thread_num >= 0);
  assert(learning_rate > 0);
  assert(sample_rate > 0);
  assert(power_freq > 0);

  size_vocabulary = vocabulary.size();
  sum_count_vocabulary = std::accumulate(count_vocabulary.begin(), count_vocabulary.end(), 0);
  n_min = corpus.size();
  for (auto &v : vocabulary) {
    const int64_t length = v.size();
    if (length < n_min) n_min = length;
  }
  for (int64_t i=0; i<size_vocabulary; i++) {
    vocabulary2id[vocabulary[i]] = i;
  }

  initialize_parameters();
  construct_unigramtable(power_freq);

  std::wcout << std::endl;
  std::wcout << "======== SCNE ========" << std::endl;
  std::wcout << "corpus.size()       : " << corpus.size() << std::endl;
  std::wcout << "vocabulary.size()   : " << vocabulary.size() << std::endl;
  std::wcout << "embed_dim           : " << embed_dim << std::endl;
  std::wcout << "random_seed         : " << random_seed << std::endl;
  std::wcout << "epoch_num           : " << epoch_num << std::endl;
  std::wcout << "neg_num             : " << neg_num << std::endl;
  std::wcout << "thread_num          : " << thread_num << std::endl;
  std::wcout << "learning_rate       : " << learning_rate << std::endl;
  std::wcout << "sample_rate         : " << sample_rate << std::endl;
  std::wcout << "power_freq          : " << power_freq << std::endl;
  std::wcout << "n_max               : " << n_max << std::endl;
  std::wcout << "n_min               : " << n_min << std::endl;
  std::wcout << "======================" << std::endl;
  std::wcout << std::endl;
}

SCNE::~SCNE() {
  delete[] embeddings_targets;
  delete[] embeddings_contexts_left;
  delete[] embeddings_contexts_right;
  delete[] table_unigram;
}

void SCNE::initialize_parameters() {
  const int64_t n = size_vocabulary*embed_dim;
  const double _min = -1.0/embed_dim;
  const double _max =  1.0/embed_dim;

  embeddings_targets        = new double[n];
  embeddings_contexts_left  = new double[n];
  embeddings_contexts_right = new double[n];

  for (int64_t i=0; i<n; i++) {
    embeddings_targets[i] = cheaprand.generate_rand_uniform(_min, _max);
    embeddings_contexts_left[i] = 0.0;
    embeddings_contexts_right[i] = 0.0;
  }
}

void SCNE::construct_unigramtable(const double power_freq) {
  table_unigram = new int64_t[SIZE_TABLE_UNIGRAM];
  double sum_count_power = 0;

  for (auto c : count_vocabulary) {
    sum_count_power += pow(c, power_freq);
  }

  int64_t id_word = 0;
  double cumsum_count_power = pow(count_vocabulary[id_word], power_freq)/sum_count_power;

  for (int64_t i_table=0; i_table<SIZE_TABLE_UNIGRAM; i_table++) {
    table_unigram[i_table] = id_word;
    if (i_table / static_cast<double>(SIZE_TABLE_UNIGRAM) > cumsum_count_power) {
      id_word++;
      cumsum_count_power += pow(count_vocabulary[id_word], power_freq)/sum_count_power;
    }
    if (id_word >= size_vocabulary) id_word = size_vocabulary - 1;
  }
}

void SCNE::train(const std::string output_base_path) {
  const int64_t length_corpus = corpus.size();
  const int64_t length_chunk = length_corpus / thread_num;
  int64_t i_corpus_start = 0;
  std::vector<std::thread> vector_threads(thread_num);

  for (int64_t id_thread=0; id_thread<thread_num; id_thread++) {
    vector_threads.at(id_thread) = std::thread(&SCNE::train_model_eachthread,
                                               this,
                                               id_thread,
                                               i_corpus_start, length_chunk, thread_num);
    i_corpus_start += length_chunk;
  }
  for (int64_t id_thread=0; id_thread<thread_num; id_thread++) {
    vector_threads.at(id_thread).join();
  }

}

void SCNE::train_model_eachthread(const int64_t id_thread,
                                  const int64_t i_corpus_start,
                                  const int64_t length_str,
                                  const int64_t thread_num)
{

  const std::wstring corpus_thread = corpus.substr(i_corpus_start, length_str);

  std::vector<int64_t> target_ngrams;
  std::unordered_map<int64_t, int64_t> ngramcounter_in_target;
  double* embeddings_contexts;
  double* target_vec = new double[embed_dim];
  std::wstring ngram;
  int64_t ngram_id, i_head_target, i_head_context, id_context;
  double coefficient = 1.0;
  double loss = 0.0;
  int64_t loss_counter = 0;

  CheapRand cheaprand_thread(id_thread + random_seed);

  if (id_thread == 0) std::wcout << std::endl;

  for (int64_t i_iteration=0; i_iteration<epoch_num; i_iteration++) {

    for (int64_t i_str=1; i_str<length_str-1; i_str++) {

      double ratio_completed = (i_iteration*(length_str-2) + i_str) / (double)(epoch_num*(length_str-2));
      if (ratio_completed > 0.9999) ratio_completed = 0.9999;
      const double _learning_rate = learning_rate * (1 - ratio_completed);

      if (id_thread == 0) {
        const int64_t i_progress = i_iteration * (length_str-2) + i_str;
        if (i_progress % SIZE_CHUNK_PROGRESSBAR == 0) {
          const double percent = 100 * (double)i_progress / (epoch_num * (length_str-2));
          std::wcout << "\rprogress : "
                     << std::fixed << std::setprecision(2) << percent << "%, loss : "
                     << std::setprecision(6) << loss/loss_counter << ", lr : "
                     << _learning_rate << " "
                     << std::flush;
          loss = 0.0;
          loss_counter = 0;
        }
      }

      target_ngrams.clear();
      ngramcounter_in_target.clear();

      for (int64_t target_length = n_min; target_length <= n_max; target_length++) {

        if (i_str+target_length > length_str) break;

        for (int64_t n = n_min; n < target_length; n++) {
          ngram = corpus_thread.substr(i_str+target_length-n, n);
          if (vocabulary2id.find(ngram) != vocabulary2id.end()) {
            ngram_id = vocabulary2id[ngram];

            if (ngramcounter_in_target.find(ngram_id) == ngramcounter_in_target.end()) {
              ngramcounter_in_target[ngram_id] = 1;
              target_ngrams.push_back(ngram_id);
            } else {
              ngramcounter_in_target[ngram_id] += 1;
            }

          }
        }
        ngram = corpus_thread.substr(i_str, target_length);
        if (vocabulary2id.find(ngram) != vocabulary2id.end()) {
          ngram_id = vocabulary2id[ngram];
          ngramcounter_in_target[ngram_id] = 1;
          target_ngrams.push_back(ngram_id);

          const int64_t freq = count_vocabulary[vocabulary2id[ngram]];
          const double probability_reject = (sqrt(freq/(sample_rate*sum_count_vocabulary)) + 1) * (sample_rate*sum_count_vocabulary) / freq;
          if (probability_reject < cheaprand_thread.generate_rand_uniform(0, 1)) continue;
        }

        std::fill_n(target_vec, embed_dim, 0);

        for (int64_t index : target_ngrams) {
          i_head_target = embed_dim * index;
          for (int64_t i=0; i<embed_dim; i++) {
            target_vec[i] += embeddings_targets[i_head_target + i];
          }
        }

        for (const bool is_right_context : {true, false}) {
          embeddings_contexts = (is_right_context) ? embeddings_contexts_right : embeddings_contexts_left;

          for (int64_t context_length=n_min; context_length <= n_max; context_length++) {
            if (is_right_context && i_str+target_length+context_length > length_str) break;
            if (!is_right_context && i_str-context_length < 0) break;
            const std::wstring context = (is_right_context) ? corpus_thread.substr(i_str+target_length, context_length) : corpus_thread.substr(i_str-context_length, context_length);

            if (vocabulary2id.find(context) == vocabulary2id.end()) break;
            id_context = vocabulary2id[context];

            for (int64_t i_ns=-1; i_ns<neg_num; i_ns++) {
              const bool is_negative_sample = (i_ns >= 0);
              if (is_negative_sample) {
                const int64_t negative_sample_index = table_unigram[cheaprand_thread.generate_randint(SIZE_TABLE_UNIGRAM)];
                if (negative_sample_index == id_context) continue;
                i_head_context = embed_dim * negative_sample_index;
              } else {
                i_head_context = embed_dim * id_context;
              }

              double inner = 0.0;
              for (int64_t i=0; i < embed_dim; i++) {
                inner += target_vec[i] * embeddings_contexts[i_head_context + i];
              }

              const double score = 1.0 / (1.0 + exp(-inner));
              const double g = score - (1.0 - (double)is_negative_sample);

              if (id_thread == 0) {
                if (!is_negative_sample) {
                  loss -= log(score + 1e-5);
                } else {
                  loss -= log(1.0 - score + 1e-5);
                }
                loss_counter += 1;
              }

              for (int64_t ngram_id : target_ngrams) {
                i_head_target = embed_dim * ngram_id;
                coefficient = static_cast<double>(ngramcounter_in_target[ngram_id]);
                for (int64_t i=0; i<embed_dim; i++) {
                  embeddings_targets[i_head_target + i] -= coefficient * _learning_rate * g * embeddings_contexts[i_head_context + i];
                }
              }
              for (int64_t i=0; i<embed_dim; i++) {
                embeddings_contexts[i_head_context + i] -= _learning_rate * g * target_vec[i];
              }
            }
          }
        }
      }
    }
  }
  if (id_thread == 0) std::wcout << std::endl << std::flush;
}


void SCNE::save_vector(const std::string output_path)
{
  std::wcout << "Save ngram vector"     << std::endl;
  std::cout << "Saving output to " << output_path << std::endl;

  std::wofstream fout(output_path);
  fout << size_vocabulary << " " << embed_dim << std::endl;
  for (int64_t i=0; i<size_vocabulary; i++) {
    fout << vocabulary[i] << " ";
    for (int64_t j=0; j<embed_dim; j++) {
      fout << embeddings_targets[i*embed_dim + j];
      if (j < embed_dim - 1) {
        fout << " ";
      }else{
        fout << std::endl;
      }
    }
  }
  fout.close();

  std::cout << "Done" << std::endl;
}
