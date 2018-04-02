//author:nicole-zhang
#include <vector>

#include "caffe/layers/kernel_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

 template <typename Dtype>
 void KernelLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  exp_value_.Reshape(bottom[0]->num(), 1, 1, 1);
  sigma_sq_ = this->layer_param_.kernel_loss_param().sigma_sq();
  inner_scale_ = (-1)/(2*sigma_sq_);
}   

template <typename Dtype>
void KernelLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int channels = bottom[0]->channels();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  for(int i=0; i<bottom[0]->num(); ++i){
      dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
  }
  caffe_cpu_scale(bottom[0]->num(), inner_scale_, dist_sq_.cpu_data(), dist_sq_.mutable_cpu_data());
  caffe_exp(bottom[0]->num(), dist_sq_.cpu_data(), exp_value_.mutable_cpu_data());

  // compute loss 
  Dtype loss(0.0); 
  for(int i=0; i<bottom[0]->num(); ++i){
    if (static_cast<int>(bottom[2]->cpu_data()[i]) == static_cast<int>(bottom[3]->cpu_data()[i])) { 
      loss += (-1)*(1-exp_value_.cpu_data()[i]);
   	} else {
      loss += 1-exp_value_.cpu_data()[i];
  	}
  }
  top[0]->mutable_cpu_data()[0] = loss/bottom[0]->num();
}

template <typename Dtype>
void KernelLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
  caffe_set(bottom[1]->count(), Dtype(0.0), bottom[1]->mutable_cpu_diff());   
  caffe_copy(diff_.count(), diff_.cpu_data(), bottom[0]->mutable_cpu_diff());
  caffe_copy(diff_.count(), diff_.cpu_data(), bottom[1]->mutable_cpu_diff());
  caffe_cpu_scale(diff_.count(), Dtype(-1.0), bottom[1]->cpu_diff(), bottom[1]->mutable_cpu_diff());
  for(int i=0; i<bottom[0]->num(); ++i){
	Dtype scale = 0;
    if (static_cast<int>(bottom[2]->cpu_data()[i]) == static_cast<int>(bottom[3]->cpu_data()[i])) {
      scale = top[0]->cpu_diff()[0]*(1/bottom[0]->num())*(-1)*(1/sigma_sq_)*exp_value_.cpu_data()[i];
    } else {
      scale = top[0]->cpu_diff()[0]*(1/bottom[0]->num())*(1/sigma_sq_)*exp_value_.cpu_data()[i];
    }
      caffe_cpu_scale(bottom[0]->channels(), scale, bottom[0]->cpu_diff()+i*bottom[0]->channels(), bottom[0]->mutable_cpu_diff()+i*bottom[0]->channels());
      caffe_cpu_scale(bottom[1]->channels(), scale, bottom[1]->cpu_diff()+i*bottom[1]->channels(), bottom[1]->mutable_cpu_diff()+i*bottom[1]->channels());
  }
}

#ifdef CPU_ONLY
STUB_GPU(KernelLossLayer);
#endif

INSTANTIATE_CLASS(KernelLossLayer);
REGISTER_LAYER_CLASS(KernelLoss);

}  // namespace caffe
