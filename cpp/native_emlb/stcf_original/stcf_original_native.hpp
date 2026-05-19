#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class StcfOriginalNative : public NativeBase {
public:
    StcfOriginalNative(int width, int height, uint64_t tau_ticks, int k, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off), tau_(tau_ticks), k_(k) { reset(); }
    void reset() override { last_ts_.assign(static_cast<size_t>(width_)*static_cast<size_t>(height_), 0); }
    py::array_t<uint8_t> accept_batch(
        py::array_t<uint64_t, py::array::c_style|py::array::forcecast> t,
        py::array_t<int32_t, py::array::c_style|py::array::forcecast> x,
        py::array_t<int32_t, py::array::c_style|py::array::forcecast> y,
        py::array_t<int8_t, py::array::c_style|py::array::forcecast> p) {
        auto tb=t.unchecked<1>(); auto xb=x.unchecked<1>(); auto yb=y.unchecked<1>(); auto pb=p.unchecked<1>();
        auto n=tb.shape(0); py::array_t<uint8_t> out(n); auto ob=out.mutable_unchecked<1>();
        for(py::ssize_t i=0;i<n;++i) ob(i)=accept_one(xb(i),yb(i),norm_pol(pb(i)),tb(i))?uint8_t{1}:uint8_t{0};
        return out;
    }
private:
    uint64_t tau_; int k_; std::vector<uint64_t> last_ts_;
    size_t idx(int32_t x,int32_t y)const noexcept{return static_cast<size_t>(y)*static_cast<size_t>(width_)+static_cast<size_t>(x);}
    bool accept_one(int32_t x,int32_t y,int8_t p,uint64_t t){
        if(!visible(x,y,p))return false;
        int cnt=0; int x0=std::max(0,x-1),x1=std::min(width_-1,x+1),y0=std::max(0,y-1),y1=std::min(height_-1,y+1);
        for(int yy=y0;yy<=y1;++yy)for(int xx=x0;xx<=x1;++xx){
            if(xx==x&&yy==y)continue;
            int64_t dt=static_cast<int64_t>(t)-static_cast<int64_t>(last_ts_[idx(xx,yy)]);
            if(0<=dt&&dt<=static_cast<int64_t>(tau_))++cnt;
        }
        last_ts_[idx(x,y)]=t; return cnt>=k_;
    }
};

} // namespace myevs_native_emlb
