#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H

#include "../basedef.h"
#include "data_point.h"
#include <cmath>

// wrapper around R's RNG such that we get a uniform distribution over
// [0,n) as required by the STL algorithm
inline int randWrapper(const int n) { return floor(unif_rand()*n); }

class data_set {
  /**
   * Collection of all data points.
   *
   * @param xpMat    pointer to bigmat if using bigmatrix
   * @param Xx       design matrix if not using bigmatrix
   * @param Yy       response values
   * @param n_passes number of passes for data
   * @param big      whether using bigmatrix or not
   * @param shuffle  whether to shuffle data set or not
   */
public:
  data_set(const SEXP& xpMat, const mat& Xx, const mat& Yy, double n_passes,
    bool big, bool shuffle) : Y(Yy), big(big), xpMat_(xpMat), shuffle_(shuffle) {
    if (!big) {
      X = Xx;
      n_samples = X.n_rows;
      n_features = X.n_cols;
    } else {
      n_samples = xpMat_->nrow();
      n_features = xpMat_->ncol();
    }
    if (shuffle_) {
      idxvec_ = std::vector<unsigned>(static_cast<unsigned>(std::ceil(n_samples*n_passes)));
      std::random_device rd;
      std::mt19937 gen(rd());

      std::uniform_int_distribution<> dis(0, n_samples - 1);

      for (unsigned i = 0; i < idxvec_.size(); ++i) {
        idxvec_[i] = dis(gen);
      }
    }
  }

  // Index to the @t th data point
  data_point get_data_point(unsigned t) const {
    t = idxmap_(t - 1);
    mat xt;
    if (!big) {
      xt = mat(X.row(t));
    } else {
      MatrixAccessor<double> matacess(*xpMat_);
      xt = mat(1, n_features);
      for (unsigned i=0; i < n_features; ++i) {
        xt(0, i) = matacess[i][t];
      }
    }
    double yt = Y(t);
    return data_point(xt, yt, t);
  }

  mat X;
  mat Y;
  bool big;
  unsigned n_samples;
  unsigned n_features;

private:
  // index to data point for each iteration
  unsigned idxmap_(unsigned t) const {
    if (shuffle_) {
      return(idxvec_[t]);
    } else {
      return(t % n_samples);
    }
  }

  Rcpp::XPtr<BigMatrix> xpMat_;
  std::vector<unsigned> idxvec_;
  bool shuffle_;
};

#endif
