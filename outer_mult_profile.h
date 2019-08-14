#include "CSC.h"
#include "CSR.h"
#include "Triple.h"
#include "radix_sort/radix_sort.hpp"
#include "utility.h"

#include <algorithm>
#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <cstring>

using namespace std;

static uint32_t nrows_per_blocker;
static uint32_t ncols_per_blocker;
static uint32_t ncols_of_A;


template <typename IT>
uint16_t fast_mod(const IT input, const int ceil) {
    return input >= ceil ? input % ceil : input;
}


template <typename IT, typename NT>
uint64_t getFlop(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    uint64_t flop = 0;

#pragma omp parallel for reduction(+ : flop)
    for (IT i = 0; i < A.cols; ++i)
    {
        IT colnnz = A.colptr[i + 1] - A.colptr[i];
        IT rownnz = B.rowptr[i + 1] - B.rowptr[i];
        flop += (colnnz * rownnz);
    }
    return flop;
}

template <typename IT, typename NT>
void do_symbolic(const CSC<IT, NT>& A, const CSR<IT, NT>& B, IT startIdx, IT endIdx,
                 uint16_t nrows_per_blocker, uint16_t ncols_per_blocker, uint16_t num_blockers,
                 IT* flop_groupby_row_blockers, IT* flop_groupby_col_blockers, IT& total_flop)
{
    #pragma omp parallel for reduction(+ : flop_groupby_row_blockers[:num_blockers]) reduction(+ : flop_groupby_col_blockers[:num_blockers*num_blockers])
    for (IT i = startIdx; i < endIdx; ++i)
    {
        IT rownnz = B.rowptr[i + 1] - B.rowptr[i];
        for (IT j = A.colptr[i]; j < A.colptr[i + 1]; ++j)
        {
            uint16_t row_blocker_id = A.rowids[j] / nrows_per_blocker;
            uint16_t col_blocker_id = fast_mod(A.rowids[j], nrows_per_blocker) / ncols_per_blocker;

            flop_groupby_row_blockers[row_blocker_id] += rownnz;
            flop_groupby_col_blockers[row_blocker_id * num_blockers + col_blocker_id] += rownnz;
        }
    }
    for (IT i = 0; i < num_blockers; ++i)
    {
        total_flop += flop_groupby_row_blockers[i];
    }
}
template <typename IT, typename NT>
bool compareTuple (tuple<IT, IT, NT> t1, tuple<IT, IT, NT> t2)
{
    if (std::get<1>(t1) != std::get<1>(t2))
        return false;
    if (std::get<0>(t1) != std::get<0>(t2))
        return false;
    return true;
}

template <typename IT, typename NT>
int64_t getReqMemory(const CSC<IT, NT>& A, const CSR<IT, NT>& B)
{
    uint64_t flop = getFlop(A, B);
    return flop * sizeof(int64_t);
}

struct ExtractKey
{
    inline int64_t operator()(tuple<int32_t, int32_t, double> tup)
    {
        int64_t res = std::get<0>(tup);
        res         = (res << 32);
        res         = res | (int64_t)(uint32_t) std::get<1>(tup);
        return res;
    }
};


struct ExtractKey2
{
    inline uint32_t operator()(tuple<int32_t, int32_t, double> tup)
    {
        return (((fast_mod(fast_mod(std::get<0>(tup), 32768), 128)) << 24) | (uint32_t) std::get<1>(tup));
    }
};

template <typename IT, typename NT>
void doRadixSort(tuple<IT, IT, NT>* begin, tuple<IT, IT, NT>* end, tuple<IT, IT, NT>* buffer)
{
    radix_sort(begin, end, buffer, ExtractKey2());
}

template <typename IT, typename NT>
IT doMerge(tuple<IT, IT, NT>* vec, IT length)
{
    if (length == 0) return 0;

    IT i          = 0;
    IT j          = 1;

    while (i < length && j < length)
    {
        if (j < length && compareTuple(vec[i], vec[j]))
            std::get<2>(vec[i]) += std::get<2>(vec[j]);
        else
            vec[++i] = std::move(vec[j]);
        ++j;
    }
    return i + 1;
}

template <typename IT>
void initializeBlockerBoundary(IT* nums_per_col_blocker, uint16_t num_blockers, IT* blocker_begin_ptr,
                               IT* blocker_end_ptr)
{
    blocker_begin_ptr[0] = 0;
    blocker_end_ptr[0]   = 0;
    for (uint16_t blocker_index = 1; blocker_index < num_blockers; ++blocker_index)
    {
        blocker_begin_ptr[blocker_index] = blocker_begin_ptr[blocker_index - 1] + nums_per_col_blocker[blocker_index - 1];
        blocker_end_ptr[blocker_index] = blocker_begin_ptr[blocker_index];
    }
}

template <typename IT, typename NT>
void OuterSpGEMM_stage(const CSC<IT, NT>& A, const CSR<IT, NT>& B, IT startIdx, IT endIdx, CSR<IT, NT>& C, \
    double &t_symbolic, double &t_pb1, double &t_pb2, double &t_sort, double &t_merge, double &t_convert, double &t_alloc, double &t_free, \
    int nblockers, int nblockchars)
{
    typedef tuple<IT, IT, NT> TripleNode;

    const uint16_t nthreads          = omp_get_max_threads();
    const uint16_t num_blockers      = nblockers;
    const uint16_t block_width       = nblockchars;
    ncols_of_A = A.cols;
    nrows_per_blocker = A.rows <= num_blockers * 64 ? 64 : (A.rows + num_blockers - 1) / num_blockers;
    ncols_per_blocker = nrows_per_blocker <= (num_blockers - 1) * 2 ? 2 : (nrows_per_blocker + num_blockers - 1) / num_blockers;

    // cout << "nrows_per_row_blocker " << nrows_per_blocker << endl;
    // cout << "ncols_per_col_blocker " << ncols_per_blocker << endl;

    IT total_nnz = 0;
    IT total_flop = 0;

    IT* row_blocker_begin_ptr = new IT[num_blockers]();
    IT* row_blocker_end_ptr   = new IT[num_blockers]();
    IT* flop_groupby_row_blockers = new IT[num_blockers]();

    IT* nnz_by_row = new IT[A.rows]();
    IT* flop_groupby_col_blockers = new IT[num_blockers * num_blockers]();

    double w1, w2;

    w1 = omp_get_wtime();
    do_symbolic(A, B, 0, A.rows, nrows_per_blocker, ncols_per_blocker, num_blockers, flop_groupby_row_blockers, flop_groupby_col_blockers, total_flop);
    w2 = omp_get_wtime();

    t_symbolic += w2 - w1;

    TripleNode* global_blockers = static_cast<TripleNode*>(operator new(sizeof(TripleNode[total_flop])));

    // calc prefix sum
    initializeBlockerBoundary(flop_groupby_row_blockers, num_blockers, row_blocker_begin_ptr, row_blocker_end_ptr);

    TripleNode* local_blockers = static_cast<TripleNode*> \
        (operator new(sizeof(TripleNode[block_width * num_blockers * nthreads])));
    IT* size_of_local_blockers = new IT[num_blockers * nthreads]();

    w1 = omp_get_wtime();

#pragma omp parallel
    {
        uint16_t thread_id = omp_get_thread_num();
// computing phase
#pragma omp for
        for (IT idx = startIdx; idx < endIdx; ++idx)
            for (IT j = A.colptr[idx]; j < A.colptr[idx + 1]; ++j) // ncols(A) * 4
            {
                IT rowid = A.rowids[j]; // nnz(A) * 4
                uint16_t row_blocker_index = rowid / nrows_per_blocker;
                IT local_blocker_size_offset = thread_id * num_blockers + row_blocker_index;
                IT local_blocker_offset = local_blocker_size_offset * block_width;
                TripleNode* cur_local_blockers = local_blockers + local_blocker_offset + size_of_local_blockers[local_blocker_size_offset];
                TripleNode* end_local_blockers = local_blockers + local_blocker_offset + block_width;
                for (IT k = B.rowptr[idx]; k < B.rowptr[idx + 1]; ++k)   // nrows(B) * 4
                {
                    *cur_local_blockers = std::move(TripleNode(rowid, B.colids[k], A.values[j] * B.values[k]));
                    cur_local_blockers++;
                    //if (size_of_local_blockers[local_blocker_size_offset] == block_width) // flop * 16
                    if (cur_local_blockers == end_local_blockers) // flop * 16
                    {
                        std::memcpy(
                            global_blockers + __sync_fetch_and_add(&row_blocker_end_ptr[row_blocker_index], block_width),
                            local_blockers + local_blocker_offset,
                            block_width * sizeof(TripleNode)
                        );
                        cur_local_blockers = local_blockers + local_blocker_offset;
                        //size_of_local_blockers[local_blocker_size_offset] = 0;

                    }
                }
                size_of_local_blockers[local_blocker_size_offset] = cur_local_blockers - local_blockers - local_blocker_offset;
            }

        for (uint16_t row_blocker_index = 0; row_blocker_index < num_blockers; row_blocker_index++)
        {
            IT local_blocker_size_offset = thread_id * num_blockers + row_blocker_index;
            IT local_blocker_offset = local_blocker_size_offset * block_width;
            std::memcpy(
                global_blockers + __sync_fetch_and_add(&row_blocker_end_ptr[row_blocker_index], size_of_local_blockers[local_blocker_size_offset]),
                local_blockers + local_blocker_offset,
                size_of_local_blockers[local_blocker_size_offset] * sizeof(TripleNode)
            );
            size_of_local_blockers[local_blocker_size_offset] = 0;
        }
    }
    w2 = omp_get_wtime();
    t_pb1 += w2 - w1;


    double FLOPS_pb1 = total_flop / (1000000000 * (w2 - w1));
    double bytes_pb1 = (A.nnz + B.nnz) * (sizeof(IT) + sizeof(NT)) + (A.cols + B.rows) * sizeof(IT) + total_flop * (2 * sizeof(IT) + sizeof(NT));
    double BW_pb1 = bytes_pb1 / (1000000000 * (w2 - w1));
    cout << "Bandwidth = " << BW_pb1 << " GB/s" <<  " GFLOPS = " << FLOPS_pb1 << endl;


    w1 = omp_get_wtime();

    vector<TripleNode*> flop_space = vector<TripleNode*>(num_blockers);
    for (uint16_t row_blocker_index = 0; row_blocker_index < num_blockers; ++row_blocker_index)
        flop_space[row_blocker_index] =
            static_cast<TripleNode*>(operator new(sizeof(TripleNode[flop_groupby_row_blockers[row_blocker_index]])));

    IT max_flops_in_col_blockers = 0;
    IT avg_flops_in_col_blockers = 0;
    IT* nnz_per_row_blocker = new IT[num_blockers]();
    IT* nnz_per_col_blocker = static_cast<IT*>(operator new(sizeof(IT[num_blockers * num_blockers])));
    IT* col_blocker_begin_ptr = static_cast<IT*>(operator new(sizeof(IT[num_blockers * num_blockers])));
    IT* col_blocker_end_ptr = static_cast<IT*>(operator new(sizeof(IT[num_blockers * num_blockers])));

    #pragma omp parallel for reduction(max : max_flops_in_col_blockers) reduction(+ : avg_flops_in_col_blockers)
    for (IT i = 0; i < num_blockers * num_blockers; ++i) {
        nnz_per_col_blocker[i] = 0;
        col_blocker_begin_ptr[i] = 0;
        col_blocker_end_ptr[i] = 0;
        avg_flops_in_col_blockers += flop_groupby_col_blockers[i];
        max_flops_in_col_blockers = max(max_flops_in_col_blockers, flop_groupby_col_blockers[i]);
    }
    TripleNode* sorting_buffer = static_cast<TripleNode*>(operator new(sizeof(TripleNode[max_flops_in_col_blockers * nthreads + 1])));

    w2 = omp_get_wtime();
    t_alloc += w2 - w1;
    // cout << "max_flops_in_col_blockers " << max_flops_in_col_blockers << " avg_flops_in_col_blockers " << double(avg_flops_in_col_blockers) / (num_blockers * num_blockers) << endl;

    w1 = omp_get_wtime();
// each thread handle a row partition
#pragma omp parallel
    {
        uint16_t thread_id = omp_get_thread_num();

#pragma omp for
        for (uint16_t row_blocker_index = 0; row_blocker_index < num_blockers; ++row_blocker_index)
        {
            IT row_base_index = row_blocker_index * num_blockers;
            initializeBlockerBoundary(flop_groupby_col_blockers + row_blocker_index * num_blockers, num_blockers, col_blocker_begin_ptr + row_base_index, col_blocker_end_ptr + row_base_index);

            for (IT rowptr = row_blocker_begin_ptr[row_blocker_index]; rowptr < row_blocker_end_ptr[row_blocker_index]; ++rowptr)
            {
                uint16_t col_blocker_index = fast_mod(std::get<0>(global_blockers[rowptr]), nrows_per_blocker) / ncols_per_blocker;
                IT local_blocker_size_offset = thread_id * num_blockers + col_blocker_index;
                IT local_blocker_offset = local_blocker_size_offset * block_width;

                local_blockers[local_blocker_offset + size_of_local_blockers[local_blocker_size_offset]++] = std::move(global_blockers[rowptr]);

                if (size_of_local_blockers[local_blocker_size_offset] == block_width)
                {
                    std::memcpy(
                        flop_space[row_blocker_index] + (col_blocker_end_ptr + row_base_index)[col_blocker_index],
                        local_blockers + local_blocker_offset,
                        block_width * sizeof(TripleNode)
                    );
                    (col_blocker_end_ptr + row_base_index)[col_blocker_index] += block_width;
                    size_of_local_blockers[local_blocker_size_offset] = 0;
                }
            }
            for (uint16_t col_blocker_index = 0; col_blocker_index < num_blockers; col_blocker_index++)
            {
                IT local_blocker_size_offset = thread_id * num_blockers + col_blocker_index;
                IT local_blocker_offset = local_blocker_size_offset * block_width;

                std::memcpy(
                    flop_space[row_blocker_index] + (col_blocker_end_ptr + row_base_index)[col_blocker_index],
                    local_blockers + local_blocker_offset,
                    sizeof(TripleNode) * size_of_local_blockers[local_blocker_size_offset]
                );
                (col_blocker_end_ptr + row_base_index)[col_blocker_index] += size_of_local_blockers[local_blocker_size_offset];
                size_of_local_blockers[local_blocker_size_offset] = 0;
            }
        }
    }
    w2 = omp_get_wtime();
    t_pb2 += w2 - w1;

    w1 = omp_get_wtime();
#pragma omp parallel
    {
        uint16_t thread_id = omp_get_thread_num();

#pragma omp for
        for (uint16_t row_blocker_index = 0; row_blocker_index < num_blockers; ++row_blocker_index)
        {
            IT row_base_index = row_blocker_index * num_blockers;
            for (uint16_t col_blocker_index = 0; col_blocker_index < num_blockers; col_blocker_index++)
            {
                doRadixSort(flop_space[row_blocker_index] + (col_blocker_begin_ptr + row_base_index)[col_blocker_index],
                            flop_space[row_blocker_index] + (col_blocker_end_ptr + row_base_index)[col_blocker_index],
                            sorting_buffer + thread_id * max_flops_in_col_blockers);
            }
        }
    }
    w2 = omp_get_wtime();
    t_sort += w2 - w1;

    w1 = omp_get_wtime();
#pragma omp parallel
    {
        uint16_t thread_id = omp_get_thread_num();

#pragma omp for
        for (uint16_t row_blocker_index = 0; row_blocker_index < num_blockers; ++row_blocker_index)
        {
            IT row_base_index = row_blocker_index * num_blockers;
            for (uint16_t col_blocker_index = 0; col_blocker_index < num_blockers; col_blocker_index++)
            {
                IT before = (col_blocker_end_ptr + row_base_index)[col_blocker_index] - (col_blocker_begin_ptr + row_base_index)[col_blocker_index];
                IT after = doMerge(flop_space[row_blocker_index] + (col_blocker_begin_ptr + row_base_index)[col_blocker_index], before);
                // col_blocker_end_ptr[col_blocker_index] = col_blocker_begin_ptr[col_blocker_index] + after;

                nnz_per_row_blocker[row_blocker_index] += after;
                nnz_per_col_blocker[row_blocker_index * num_blockers + col_blocker_index] = after;
                __sync_fetch_and_add(&total_nnz, after);
            }

        } // outer-most row-wise for loop
    } // outer-most parellel block

    w2 = omp_get_wtime();
    t_merge += w2 - w1;

    w1 = omp_get_wtime();

    IT *cumulative_colid_indices = new IT[num_blockers * num_blockers + 1]();
    IT *cumulative_col_blocker_indices = new IT[num_blockers * num_blockers + 1]();
    scan(nnz_per_col_blocker, cumulative_colid_indices, (IT)(num_blockers * num_blockers));

    if (C.isEmpty())
    {
        C.make_empty();
    }
    C.rows = A.rows;
    C.cols = B.cols;

    C.colids = static_cast<IT*>(operator new(sizeof(IT[total_nnz])));
    C.values = static_cast<NT*>(operator new(sizeof(NT[total_nnz])));
    C.rowptr = static_cast<IT*>(operator new(sizeof(IT[C.rows + 1])));

    C.rowptr[0] = 0;

    #pragma omp parallel for
    for (uint16_t row_blocker_index = 0; row_blocker_index < num_blockers; ++row_blocker_index)
        for (uint16_t col_blocker_index = 0; col_blocker_index < num_blockers; col_blocker_index++)
            {
                scan(flop_groupby_col_blockers + row_blocker_index * num_blockers, cumulative_col_blocker_indices + row_blocker_index * num_blockers, (IT)(num_blockers));
                IT base = cumulative_colid_indices[row_blocker_index * num_blockers + col_blocker_index];
                auto space_addr = flop_space[row_blocker_index] + cumulative_col_blocker_indices[row_blocker_index * num_blockers + col_blocker_index];

                for (IT index = 0; index < nnz_per_col_blocker[row_blocker_index * num_blockers + col_blocker_index]; ++index)
                {
                    ++nnz_by_row[std::get<0>(space_addr[index])];
                    C.colids[base + index] = std::get<1>(space_addr[index]);
                    C.values[base + index] = std::get<2>(space_addr[index]);
                }
            }
    scan(nnz_by_row, C.rowptr, C.rows + 1);
    C.nnz = total_nnz;
    w2 = omp_get_wtime();
    t_convert += w2 - w1;

    w1 = omp_get_wtime();
    my_free<TripleNode>(global_blockers);
    my_free<TripleNode>(local_blockers);
    my_free<IT>(size_of_local_blockers);
    my_free<IT>(row_blocker_begin_ptr);
    my_free<IT>(row_blocker_end_ptr);
    my_free<IT>(flop_groupby_row_blockers);
    my_free<IT>(flop_groupby_col_blockers);
    my_free<IT>(nnz_by_row);
    my_free<IT>(nnz_per_row_blocker);
    my_free<IT>(nnz_per_col_blocker);
    my_free<IT>(col_blocker_begin_ptr);
    my_free<IT>(col_blocker_end_ptr);
    my_free<IT>(cumulative_colid_indices);
    my_free<IT>(cumulative_col_blocker_indices);
    for (uint16_t row_blocker_index = 0; row_blocker_index < num_blockers; ++row_blocker_index)
        my_free<TripleNode>(flop_space[row_blocker_index]);
    w2 = omp_get_wtime();
    t_free += w2 - w1;
}


template <typename IT, typename NT>
void OuterSpGEMM(const CSC<IT, NT>& A, const CSR<IT, NT>& B, CSR<IT, NT>& C, int nblockers, int nblockchars)
{
    double t_symbolic = 0;
    double t_pb1 = 0;
    double t_pb2 = 0;
    double t_sort = 0;
    double t_merge = 0;
    double t_convert = 0;
    double t_alloc = 0;
    double t_free = 0;

    int niter = 5;

    for (int i = 0; i < niter; ++i) {
        OuterSpGEMM_stage(A, B, 0, A.rows, C, t_symbolic, t_pb1, t_pb2, t_sort, t_merge, t_convert, t_alloc, t_free, nblockers, nblockchars);
        // cout << "Iter " << i << " Completed!" << endl;
    }
    cout << "symbolic took " << t_symbolic / niter << endl;
    cout << "PB1 took " << t_pb1 / niter << endl;
    cout << "PB2 took " << t_pb2 / niter << endl;
    cout << "Sort took " << t_sort / niter << endl;
    cout << "Merge took " << t_merge / niter << endl;
    cout << "Convert took " << t_convert / niter << endl;
    // cout << "Alloc took " << t_alloc / niter << endl;
    // cout << "Free took " << t_free / niter << endl << endl;;
}
