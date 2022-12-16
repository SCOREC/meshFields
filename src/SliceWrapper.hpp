#ifndef slicewrapper_hpp
#define slicewrapper_hpp

#include <Cabana_Core.hpp>

namespace SliceWrapper {

template <class SliceType, class T> struct SliceWrapper {

  SliceType slice;

  typedef T Type;

  SliceWrapper(SliceType slice_in) : slice(slice_in) {}

  SliceWrapper() {}

  /* access functions
     
     The access functions are used to get a specific element from a Field. 
     If the user creates a field of an array, the third argument will be the index of the array.
     If the user craetes a field of a 2d array, the third and fourth argument will indexes the two arrays, and so on
  */
  
  KOKKOS_INLINE_FUNCTION
  T &access(int s, int a) const { return st_.access(s, a); }
  
  KOKKOS_INLINE_FUNCTION
  auto &access(int s, int a, int i) const { return st_.access(s, a, i); }

  KOKKOS_INLINE_FUNCTION
  auto &access(int s, int a, int i, int j) const {
    return st_.access(s, a, i, j);
  }

  KOKKOS_INLINE_FUNCTION
  auto &access(int s, int a, int i, int j, int k) const {
    return st_.access(s, a, i, j, k);
  }
};

using namespace Cabana;

template <class ExecutionSpace, class MemorySpace, class... Ts>
class CabSliceController {

  // type definitions
  using TypeTuple = std::tuple<Ts...>;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<Ts...>;

public:
  
  static constexpr int vecLen =
      Cabana::AoSoA<DataTypes, DeviceType>::vector_length;

private:

  // all the type defenitions that are needed us to get the type of the slice returned by the underlying AoSoA
  using soa_t = SoA<DataTypes, vecLen>;

  template <std::size_t index>
  using member_data_t =
      typename Cabana::MemberTypeAtIndex<index, DataTypes>::type;

  template <std::size_t index>
  using member_value_t =
      typename std::remove_all_extents<member_data_t<index>>::type;

  template <class T, int stride>
  using member_slice_t =
      Cabana::Slice<T, DeviceType, Cabana::DefaultAccessMemory, vecLen, stride>;

  template <class T, int stride>
  using wrapper_slice_t = SliceWrapper<member_slice_t<T, stride>, T>;

  // member vaiables
  Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa;
  const int num_tuples;

public:
  // a tool that allows a single index to be converted into two indexes that can be used for slice traversal
  struct IndexToSA {
    IndexToSA(int vecLen_in) : vecLen(vecLen_in) {}
    int vecLen;
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, int &s, int &a) const {
      s = i / vecLen;
      a = i % vecLen;
    }
  };

  IndexToSA indexToSA;

  // constructors (default constructor necessary)
  
  CabSliceController() {}

  CabSliceController(int n) : aosoa("sliceAoSoA", n), num_tuples(n), indexToSA(aosoa.vector_length) {}

  // size function to get the number of tuples
  
  int size() { return num_tuples; }

  /* make a slice of the underlying AoSoA based on a user specified index
    
     get the type of the value in the slice at the index
     calculate the 'stride' value, needed for construction of the slice
     create the slice
     return the wrapped slice  
  */
  
  template <std::size_t index> auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    const int stride = sizeof(soa_t) / sizeof(member_value_t<index>);
    auto slice = Cabana::slice<index>(aosoa);
    return wrapper_slice_t<type, stride>(std::move(slice));
  }
  
  /* Parallel functions
     
     These functions run the given kernel in parallel on the GPU
     (unless the execution space is serial)
     parallel_for - a for loop that will iterate from lowerBound to upperbound
     parallel_reduce - a way for a user to pass in a reduction kernel and a reducer to make their own reductions
  */

  template <typename FunctorType, typename ReductionType>
  void parallel_reduce(FunctorType &reductionKernel, ReductionType &reductionType, std::string tag) {
    Kokkos::RangePolicy<ExecutionSpace> policy(0, num_tuples);
    Kokkos::parallel_reduce(tag, policy, reductionKernel, reductionType);
  }

  template <typename FunctorType>
  void parallel_for(int lowerBound, int upperBound, FunctorType &vectorKernel, std::string tag) {
    Cabana::SimdPolicy<vecLen, ExecutionSpace> simdPolicy(lowerBound, upperBound);
    Cabana::simd_parallel_for(simdPolicy, vectorKernel, tag);
  }
  
};

} // namespace SliceWrapper

#endif
