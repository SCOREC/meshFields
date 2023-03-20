#ifndef meshfield_hpp
#define meshfield_hpp

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cstdio>
#include <type_traits> // std::same_v<t1,t2>
#include <array>

#include "MeshField_Utility.hpp"

//#include "CabanaSliceWrapper.hpp"

namespace MeshField {

template <class Slice> class Field {

  Slice slice;
  typedef typename Slice::Type Type;

public:

  Field(Slice s) : slice(s) {}
  
  /* operator() -> 1D Access */
  /* access() -> 2D Access */

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s) const { return slice(s); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a) const { return slice(s, a); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i) const { return slice(s, a, i); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i, int j) const {
    return slice(s, a, i, j);
  }
  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i, int j, int k) const {
    return slice(s, a, i, j, k);
  }

};

template <class Controller> class MeshField {

  Controller sliceController;

public:

  MeshField(Controller controller) : sliceController(std::move(controller)) {}

  template <std::size_t index> auto makeField() {
    auto slice = sliceController.template makeSlice<index>();
    return Field(std::move(slice));
  }
  
  template <class Field, class View> void setField(Field &field, View &view) {
    /*
    auto view_rank = view::rank + view::rank_dynamic;
    auto field_rank = Kokkos::View<field::ty>::rank + Kokkos::View<field::type>::rank_dynamic;
    assert(view_rank == field_rank);
    
    if( view_rank <= 1 ) {
      Kokkos::RangePolicy<> p(0,view.size()); 
      Kokkos::parallel_for()
    } else {
    
    }
    */
  }
  
  /*
  template <class FieldType, class T = typename FieldType::Type>
  T sum(FieldType &field) {
    T result;
    auto indexToSA = sliceController.indexToSA;
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lsum) {
      int s, a;
      indexToSA(i, s, a);
      lsum += field(s, a);
    };
    sliceController.parallel_reduce(reductionKernel, result, "sum_reduce");
    return result;
  }

  template <class FieldType, class T = typename FieldType::Type>
  T min(FieldType &field) {
    T min;
    auto indexToSA = sliceController.indexToSA;
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lmin) {
      int s, a;
      indexToSA(i, s, a);
      lmin = lmin < field(s, a) ? lmin : field(s, a);
    };
    auto reducer = Kokkos::Min<T>(min);
    sliceController.parallel_reduce(reductionKernel, reducer, "min_reduce");
    return min;
  }

  template <class FieldType, class T = typename FieldType::Type>
  T max(FieldType &field) {
    T max;
    auto indexToSA = sliceController.indexToSA;
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lmax) {
      int s, a;
      indexToSA(i, s, a);
      lmax = lmax > field(s, a) ? lmax : field(s, a);
    };
    auto reducer = Kokkos::Max<T>(max);
    sliceController.parallel_reduce(reductionKernel, reducer, "max_reduce");
    return max;
  }

  template <class FieldType> double mean(FieldType &field) {
    return static_cast<double>(sum(field)) / sliceController.size();
  }
  */

  template <typename FunctorType, class IS, class IE>
  void parallel_for(const std::initializer_list<IS>& start, 
                    const std::initializer_list<IE>& end,
                    FunctorType &vectorKernel,
                    std::string tag) {
    sliceController.parallel_for(start,end, vectorKernel, tag);
  }
  
  
  template <typename FunctorType, class ReducerType, class IS, class IE>
  void parallel_reduce(const std::initializer_list<IS>& start,
                       const std::initializer_list<IE>& end,
                       FunctorType &reductionKernel, ReducerType &reducer,
                       std::string tag) {
    /* RUN IT HERE */
    /* TODO: Apply executionspace from controller to policy. */
    /* TODO: infinite reducers */
    constexpr auto RANK = MeshFieldUtil::function_traits<FunctorType>::arity;
    assert( start.size() == end.size() );
    if constexpr ( RANK <= 1 ) {
      Kokkos::RangePolicy</*TODO*/> policy((*start.begin()), (*end.begin()) );
      Kokkos::parallel_reduce( tag, policy, reductionKernel, reducer );
    } else {
      Kokkos::Array<int64_t,RANK> a_start = MeshFieldUtil::to_kokkos_array( start );
      Kokkos::Array<int64_t,RANK> a_end = MeshFieldUtil::to_kokkos_array( end );
      Kokkos::MDRangePolicy<Kokkos::Rank<RANK> /*TODO*/> policy(a_start, a_end);
      Kokkos::parallel_reduce( tag, policy, reductionKernel, reducer );
    }

  }
  
  /*
  template <typename KernelType>
  void parallel_scan(KernelType &scanKernel, std::string tag) {
    Kokkos::parallel_scan(tag, sliceController.size() + 1, scanKernel);
  }
  // depending on size of dimensions, take variable number of arguements
  // that give pairs of lower and upper bound for the multi-dim views.
  */
};

} // namespace MeshField

#endif
