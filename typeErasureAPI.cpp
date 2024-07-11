
#include <iostream>
#include <memory>
#include <vector>

/*
The issue
--------------------------------------------------------------------------------------------------------------------------------
We want common or nearly common interfaces.

Consider an increment() function:
Incrementing a matrixField or a scalerField may require different arguments
-We need to pass both these kinds of arguments!

Essentially if some operations require different arguments depending on the field they are applied to,
this makes it hard to pass one argument to each entry in, say, a vector.


What functions do we even need?
--------------------------------------------------------------------------------------------------------------------------------
-Draw fields : print metadata of field (insensitive to args which is nice)
-
-



What does the user potentially want to do at the highest level that we need to handle?
--------------------------------------------------------------------------------------------------------------------------------
Sycronize fields (one process changed something and the others need to update accordingly)


Since type erasure is the design pattern we have chosen, how can we recover the type?
2 cases
-We want to do the same operation on everything in a container 
 (eg loop through and print each indicidual field's data) (no sensitivity to kind of field)
-User needs specific access to object in container and wants to do specific operation depending on type of object



Motivation for type erasure
--------------------------------------------------------------------------------------------------------------------------------
Inheritance requires pointer semantics which isn't modern cpp :(
Don't do that, it may be in a lot of code but the current language is moving away from that
So type erasure designed in a way that Field is closed for modification is good
(Will have to make note for user what new kinds of field implementations require eg someField)


TO DO
--------------------------------------------------------------------------------------------------------------------------------

do_size(): print rank / extent of each field, total

for the scorec team Field will be open to modification for a while
so dont worry about getting everything now

make write up to explain 
- how to create a new "do_" function
- requirements of a new type of field if user wants to implement
- basic use of the api, how to perform certain use cases
       - maybe even make some test code to show off the features

Add namespace around the kinds of fields we have in this code eg: meshfields::kokkos
to distinguish between kokkos and cabana implementations

*/


class ScalarField
{ 
    public:
        explicit ScalarField( /*FieldEntityStorage storage, Mesh mesh*/){

            // call createRealStorage_kk
            // which should be modified to return the kokkos controller it creates
            // from temp or some such function?
        }

        void getInfo() const noexcept;
        void scale(){ std::cout << "scalarField: " << __func__ << "\n"; }
        // Other universal functions...   

    private:
        double someMember;
        // ... Remaining data members
};

class MatrixField
{
    public:
        explicit MatrixField( /*FieldEntityStorage storage, Mesh mesh*/ ){

            // this should also call createRealStorage_kk
            // once it is modified to be templated to take any kind of data
            // and not just defaulted to doubles
        }

        void getInfo() const noexcept;
        void scale(){ std::cout << "matrixField: " << __func__ << "\n"; }
        // Other universal functions... 

        // Functions that must take different arguments
        // eg: scaling a matrix vs vector
        // (any reason not to consider these unique functions, can we universalize a do_ ?)

        // unique functions 

    private:
        double someMember;
        // ... Remaining data members
};



// Need a fucntion that returns FieldT (the type in a usuable form, cant return a type)
// auto field = fieldwrapper.get(){return &field_}
// ctrl f for: "see this thing here"
// returning a refrence only

// by giving user the ability to label each Field with a string
// and giving the user the abiltiy to grab a refrence to the underlying field
// (essentially taking the thing out of the box)

// refrence is always fixed and immutable
// we need both:
// Const refrence; can only call const methods of object
// Non const refrence; no restrictions

// this allows the user to apply field specific functions


class Field
{
    private:
        struct FieldConcept
        {
            virtual ~FieldConcept() = default; 

            virtual void do_scale( /*...*/ ) = 0;
  
            virtual std::unique_ptr<FieldConcept> clone() const = 0;

            // More functions...
        };


        template< typename FieldT > 
        struct FieldModel : public FieldConcept
        {
            // Constructor take ownership of some kind of field e.g. scalar, matrix, etc
            FieldModel( FieldT field )
                : field_{ std::move(field) }
            {}

            std::unique_ptr<FieldConcept> clone() const override
            {
                return std::make_unique<FieldModel>(*this);
            }

            // do_ prefix to avoid naming conflicts
            // This may not be in the best place. The point of a do_ function is to allow us
            // to call functions that take the same arguments regardless of the type of field
            // so we don't have to "look in the box". It lets us loop through a container of
            // fields more easily at the cost of requiring each field in the container to have 
            // an implementation for that specific function.
            void do_scale( /**/ ) override
            {
                field_.scale();
                // Alternative implementation calls a friend function
            }

            FieldT field_; // "see this thing here"
            std::string label; // Where is the best place to assign this? Should it go in Field instead of here? Assign vs store
        };

        std::unique_ptr<FieldConcept> pimpl; // pointer to implementation

    public:

        template< typename FieldT >
        Field( FieldT field )
            : pimpl{ std::make_unique<FieldModel<FieldT>>( std::move(field) ) }
        {}

        void scale() {
          pimpl->do_scale();
        };

        /* Not important right now, focus on basics
        // Copy operations
        Field( Field const& other )
            : pimpl( other.pimpl->clone() )
        {}

        Field& operator=( Field const& other )
        {
            // copy-and-swap idiom
            other.pimpl->clone().swap( pimpl );
            return *this;
        }
        */
        // Move operations, !!!!! see 22:00 ish to understand the issues
        //Field( Field&& other ) = default;
        //Field& operator=( Field&& other ) = default;

        //...
};

int main()
{
    using Fields = std::vector<Field>;

    // Creating some fields
    Fields fields;
    fields.emplace_back( ScalarField{ /**/ } ); 
    fields.emplace_back( MatrixField{ /**/ } ); 
    for (auto& f : fields) f.scale();
}



