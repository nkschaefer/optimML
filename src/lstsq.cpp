#include <string>
#include <algorithm>
#include <vector>
#include <iterator>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <functional>
#include <cstdlib>
#include <utility>
#include <lapack.h>
#include "nnls/nnls.h"

using std::cout;
using std::endl;
using namespace std;

// ===== lstsq =====
// Functions related to least-squares solutions of the
// form Ax=b.

/**
 * Solve a linear system of the form Ax = b
 * 
 * Optionally incorporate weights
 * 
 * This function is not exposed in the header file; users should
 * call the overloaded function lstsq() instead, which calls this.
 *
 * These were useful references:
 *
 * https://netlib.org/lapack/explore-html/d0/db8/group__real_g_esolve_gabc655f9cb0f6cfff81b3cafc03c41dcb.html
 * https://github.com/numpy/numpy/blob/v1.11.0/numpy/linalg/linalg.py#L1785-L1943
 * https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix
 */
bool lapack_weighted_lstsq(vector<vector<double> >& a,
    vector<double>& y,
    vector<double>& weights,
    vector<double>& result_coefficients,
    bool use_weights){

    int M = a.size(); // rows of A
    int N = a[0].size(); // cols of A
    int NRHS = 1; // cols of X
    int LDA = M; // min(1, M)
    int LDB = max(N,M); // rows of Y
    
    if (M != y.size()){
        fprintf(stderr, "dimensions do not match\n");
        exit(1);
    }
    int info;
    int lwork = -1; // automatically allocate workspace size
    int rank = 0;
    
    // Negative uses machine precision, otherwise singular values 
    // S(i) <= rcond * S(1) are treated as zero
    double rcond = -1.0;
    double wkopt;
    double* work;
    
    /*
        IWORK is INTEGER array, dimension (MAX(1,LIWORK))
        LIWORK >= max(1, 3*MINMN*NLVL + 11*MINMN),
        where MINMN = MIN( M,N ).
        
        Seems from other documentation like "smlsiz" can be 25.
    */
    int smlsiz = 25;
    int nlvl = max(0,(int)round(log2((float)min(M,N)/(float)(smlsiz+1)))+1);
    int iworkdims = 3*min(M,N)*nlvl + 11*min(M,N);
    int iwork[iworkdims];
    
    int sdim = min(M,N);
    double s[sdim];
    // Populate A matrix
    // NOTE: entries are in order of columns, then rows
    
    double* a_lapack = new double[M*N];
    
    int k = 0;
    for (int j = 0; j < N; ++j){
        for (int i = 0; i < M; ++i){
            if (isnan(a[i][j]) || isinf(a[i][j])){
                fprintf(stderr, "ERROR: A[%d][%d] is invalid\n", i, j);
                return false;
            }
            // Here's where the weights come in: multiply A entry
            // by sqrt of corresponding row of weight vector
            if (use_weights){
                a_lapack[i+j*M] = a[i][j] * sqrt(weights[i]);
            }
            else{
                a_lapack[i+j*M] = a[i][j];
            }
        }
    }
    // Populate B (y) matrix
    double* b_lapack = new double[LDB*NRHS];
    for (int i = 0; i < y.size(); ++i){
        if (isnan(y[i]) || isinf(y[i])){
            fprintf(stderr, "ERROR: b[%d] is invalid\n", i);
            return false;
        }
        if (use_weights){
            // Incorporate weight into B vector
            b_lapack[i] = y[i] * sqrt(weights[i]);
        }
        else{
            b_lapack[i] = y[i];
        }
    }
    // First command queries and allocates workspace
    LAPACK_dgelsd(&M, 
        &N, 
        &NRHS, 
        a_lapack, 
        &LDA, 
        b_lapack, 
        &LDB, 
        s, 
        &rcond, 
        &rank, 
        &wkopt, 
        &lwork,
        iwork, 
        &info);
    lwork = (int)wkopt;
    work = (double*)malloc(lwork*sizeof(double));
    // Second command finds solution
    LAPACK_dgelsd(&M, 
        &N, 
        &NRHS, 
        a_lapack, 
        &LDA, 
        b_lapack, 
        &LDB, 
        s, 
        &rcond, 
        &rank, 
        work, 
        &lwork,
        iwork, 
        &info);
    if (info > 0){
        fprintf(stderr, "Least squares did not converge\n");
        return false;
    }
    
    if (result_coefficients.size() < a[0].size()){
        for (int i = 0; i < a[0].size(); ++i){
            result_coefficients.push_back(0.0);
        }
    }
    for (int i = 0; i < a[0].size(); ++i){
        result_coefficients[i] = b_lapack[i];
    }
    free(work);
    delete[] a_lapack;
    delete[] b_lapack; 
    return true;
}

/**
 * Same as above, but uses non-negative least squares
 * algorithm (no x entries can be below zero)
 *
 */
bool weighted_nn_lstsq(vector<vector<double> >& a,
    vector<double>& b,
    vector<double>& weights,
    vector<double>& result_coefficients,
    bool use_weights){
    
    int M = a.size(); // rows of A
    int N = a[0].size(); // cols of A
    int NRHS = 1; // cols of X
    int MDA = M; // min(1, M)
    int LDB = max(N,M); // rows of Y
    
    // Store results
    double* x = new double[N];
    
    // Allocate workspace variables    
    double* work = new double[N];
    double* zz = new double[M];
    int* index = new int[2*N];
    if (M != b.size()){
        fprintf(stderr, "dimensions do not match\n");
        exit(1);
    }
    
    // Populate A matrix
    // NOTE: entries are in order of columns, then rows
    double* a_lapack = new double[M*N];
    int k = 0;
    for (int j = 0; j < N; ++j){
        for (int i = 0; i < M; ++i){
            // Here's where the weights come in: multiply A entry
            // by sqrt of corresponding row of weight vector
            if (use_weights){
                a_lapack[i+j*M] = a[i][j] * sqrt(weights[i]);
            }
            else{
                a_lapack[i+j*M] = a[i][j];
            }
        }
    }
    // Populate B (y) matrix
    double* b_lapack = new double[LDB*NRHS];
    for (int i = 0; i < b.size(); ++i){
        if (use_weights){
            // Incorporate weight into B vector
            b_lapack[i] = b[i] * sqrt(weights[i]);
        }
        else{
            b_lapack[i] = b[i];
        }
    }
    int mode;
    double residual;
    int ret = nnls_c(a_lapack, &MDA, &M, &N, b_lapack, x, &residual,
        work, zz, index, &mode);
    if (mode == 3){
        fprintf(stderr, "NNLS did not converge\n");
        return false;
    }
    else if (mode == 2){
        fprintf(stderr, "NNLS: bad dimensions\n");
        return false;
    }
    if (result_coefficients.size() < a[0].size()){
        result_coefficients.clear();
        for (int i = 0; i < a[0].size(); ++i){
            result_coefficients.push_back(0.0);
        }
    }
    for (int i = 0; i < a[0].size(); ++i){
        result_coefficients[i] = x[i];
    }
    delete[] a_lapack;
    delete[] b_lapack; 
    delete[] x;
    delete[] work;
    delete[] zz;
    delete[] index;
    return true;
}

/**
 * Solve the linear system Ax=b without weights 
 * for rows of A.
 */
bool lstsq(vector<vector<double> >& A,
    vector<double>& b,
    vector<double>& x){
    
    // No weight vector.
    vector<double> dummy;
    bool success = lapack_weighted_lstsq(A, b, dummy,
        x, false);
    return success;
}

/**
 * Solve the linear system Ax=b with weights for
 * rows of A.
 */
bool lstsq(vector<vector<double> >& A,
    vector<double>& b,
    vector<double>& weights,
    vector<double>& x){

    // Has weight vector.
    bool success = lapack_weighted_lstsq(A, b, weights,
        x, true);
    return success;
}

/**
 * Solve the linear system Ax=b without weights
 * for rows of A, and requiring all x to be positive.
 */
bool nn_lstsq(vector<vector<double> >& A,
    vector<double>& b,
    vector<double>& x){
    
    // No weight vector.
    vector<double> dummy;
    bool success = weighted_nn_lstsq(A, b, dummy, x,
        false);
    return success;
}

/**
 * Solve the linear system Ax=b with weights 
 * for rows of A, and requiring all x to be positive.
 */
bool nn_lstsq(vector<vector<double> >& A,
    vector<double>& b,
    vector<double>& weights,
    vector<double>& x){
    
    // Has weight vector.
    bool success = weighted_nn_lstsq(A, b, weights, x, true);
    return success;
}

