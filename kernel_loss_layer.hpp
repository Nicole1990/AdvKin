//author:nicole-zhang
#ifndef CAFFE_KERNEL_LOSS_LAYER_HPP_
#define CAFFE_KERNEL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class KernelLossLayer : public LossLayer<Dtype> {
 public:
  explicit KernelLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
//   virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);
      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
//  virtual inline int MinTopBlobs() const { return 1; }
 // virtual inline int MaxTopBlobs() const { return 3; }
   virtual inline int ExactNumTopBlobs() const { return 1; }
   virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline const char* type() const { return "KernelLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      //const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> dist_sq_;
  Blob<Dtype> exp_value_;
  Dtype sigma_sq_;
  Dtype inner_scale_;
};

}  // namespace caffe

#endif  // CAFFE_KERNEL_LOSS_LAYER_HPP_
