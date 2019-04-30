#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>


//#include "overridenew.h"
#include "../utility.h"
#include "../CSC.h"
#include "sample_common.hpp"
#include "../multiply.h"

using namespace std;


#define VALUETYPE double
#define INDEXTYPE int32_t


template <typename IT, typename NT>
void mtxstream(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    IT* a             = A.rowids;
    IT* b             = B.colids;
    double start_time = omp_get_wtime();
    int niter         = 10;
    for (int iter = 0; iter < niter; ++iter)
    {
        for (IT i = 0; i < A.cols; ++i)  // outer product of ith row of A and ith column of B
        {
            // IT rownnz = B.rowptr[i+1] - B.rowptr[i];
            // IT colnnz = A.colptr[i+1] - A.colptr[i];
            // total_flop += (colnnz * rownnz);
            // total_flop += (B.rowptr[i] - A.colptr[i]);
            IT start = A.colptr[i];
            IT end   = A.colptr[i + 1];
            for (IT j = start; j < end; ++j)  // For all the nonzeros of the ith column
            {
                a[j] = b[j];
            }
        }
    }

    double end_time = omp_get_wtime();
    double msec     = ((end_time - start_time) * 1000) / niter;
    double N        = A.nnz + A.rows;
    int itemsize    = sizeof(IT);

    double bandwidth = 2 * (double) N * itemsize / 1024 / 1024 / msec;
    cout << "bandwidth : " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << endl;
}

template <typename NT>
void stream(vector<NT> a, vector<NT> b, int itemsize)
{
    int64_t N    = a.size();
    double start = omp_get_wtime();
    int niter    = 1000;


    for (int iter = 0; iter < niter; ++iter)
    {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            a[i] = b[i];
        }
    }
    double end  = omp_get_wtime();
    double msec = ((end - start) * 1000) / niter;

    double bandwidth = 2 * (double) N * itemsize / 1024 / 1024 / msec;
    cout << "StreamTest : " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << endl;
}

template <typename IT, typename NT>
void StreamTest(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    vector<IT> stream1(A.nnz, 0);
    std::iota(stream1.begin(), stream1.end(), 0);
    vector<IT> stream2(A.nnz, 0);
    std::iota(stream2.begin(), stream2.end(), 0);
    stream(stream1, stream2, sizeof(IT));
}


int main(int argc, char* argv[])
{
    vector<int> tnums = {1};

	CSC<INDEXTYPE, VALUETYPE> A_csc, B_csc, C_csc_corret;


	if (argc < 4) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|ts|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
        return -1;
    }
    else if (argc < 6) {
        cout << "Normal usage: ./spgemm {gen|binary|text} {rmat|er|ts|matrix1.txt} {scale|matrix2.txt} {edgefactor|product.txt} <numthreads>" << endl;
    }
    else {
        cout << "Running on " << argv[5] << " processors" << endl << endl;
        tnums = {atoi(argv[5])};
    }

    /* Generating input matrices based on argument */
    SetInputMatricesAsCSC(A_csc, B_csc, argv);

    CSR<INDEXTYPE, VALUETYPE> B_csr (B_csc);

  	A_csc.Sorted();
  	B_csc.Sorted();
    B_csr.Sorted();

    double start, end, msec, ave_msec, mflops;
    auto nfop = get_flop(A_csc, B_csr);

    for (int tnum : tnums) {
        omp_set_num_threads(tnum);
        StreamTest(A_csc, B_csr);
    }

    A_csc.make_empty();
    B_csc.make_empty();
    B_csr.make_empty();

    return 0;
}
