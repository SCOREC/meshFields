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
  auto size(int i) const { return slice.size(i); }

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
  
  int size(int type_index, int dimension_index) 
  { return sliceController.size(type_index,dimension_index); }

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
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lsum) {
      lsum += field(i);
    };
    sliceController.parallel_reduce(reductionKernel, result, "sum_reduce");
    return result;
  }
  
  template <class FieldType, class T = typename FieldType::Type>
  T min(FieldType &field) {
    T min;
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lmin) {
      lmin = lmin < field(i) ? lmin : field(i);
    };
    auto reducer = Kokkos::Min<T>(min);
    this->parallel_reduce(reductionKernel, reducer, "min_reduce");
    return min;
  }

  template <class FieldType, class T = typename FieldType::Type>
  T max(FieldType &field) {
    T max;
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lmax) {
      lmax = lmax > field(i) ? lmax : field(i);
    };
    auto reducer = Kokkos::Max<T>(max);
    this->parallel_reduce("max_reduce", reductionKernel, reducer, "max_reduce");
    return max;
  }
  */
  /*
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
  
  template <typename FunctorType, class IS, class IE, class ReducerType>
  void parallel_reduce(std::string tag,
                       const std::initializer_list<IS>& start,
                       const std::initializer_list<IE>& end,
                       FunctorType &reductionKernel,
                       ReducerType &reducer) {
    /* TODO: infinite reducers */
    /* Number of arguements to lambda should be equal to number of ranks + number of reducers
     * -> adjust 'RANK' accordingly */
    constexpr std::size_t reducer_count = 1;
    constexpr auto RANK = MeshFieldUtil::function_traits<FunctorType>::arity - reducer_count; 
    
    using EXE_SPACE = typename Controller::exe;
    assert( start.size() == end.size() );
    if constexpr ( RANK <= 1 ) {
      Kokkos::RangePolicy<EXE_SPACE> policy((*start.begin()), (*end.begin()) );
      Kokkos::parallel_reduce( tag, policy, reductionKernel, reducer );
    } else {
      auto a_start = MeshFieldUtil::to_kokkos_array<RANK>( start );
      auto a_end = MeshFieldUtil::to_kokkos_array<RANK>( end );
      Kokkos::MDRangePolicy<Kokkos::Rank<RANK>, EXE_SPACE> policy(a_start, a_end);
      Kokkos::parallel_reduce( tag, policy, reductionKernel, reducer );
    }

  }
  //TODO TEST
  template <typename KernelType>
  void parallel_scan(std::string tag,
                     int64_t start_index,
                     int64_t end_index, 
                     KernelType &scanKernel ) {
    Kokkos::RangePolicy p(start_index, end_index);
    Kokkos::parallel_scan(tag, p, scanKernel);
  }
  // depending on size of dimensions, take variable number of arguements
  // that give pairs of lower and upper bound for the multi-dim views.
};

} // namespace MeshField

#endif
