#ifndef CHEAPRAND_H
#define CHEAPRAND_H

class CheapRand {

private:
  uint64_t randomstate;

public:
  CheapRand() {
    randomstate = 0;
  }

  explicit CheapRand(int64_t _seed) {
    assert(_seed >= 0);
    randomstate = _seed;
  }

  CheapRand(CheapRand& cheaprand) {
    randomstate = cheaprand.get_randomstate();
  }

  int64_t get_randomstate() {
    return randomstate;
  }

  inline int64_t generate_randint(const int64_t max) {
    assert(max > 0);
    randomstate = randomstate * 25214903917 + 11;
    return std::abs(static_cast<int64_t>(randomstate >> 16)) % max;
  }

  inline double generate_rand_uniform(const double _min, const double _max) {
    return _min + (_max - _min) * generate_randint(65536) / static_cast<double>(65536);
  }

};
#endif
