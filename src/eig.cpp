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

using std::cout;
using std::endl;
using namespace std;

// ===== eig =====
// Functions related to eigen decomposition of matrices.

/**
 * Eigen-decompose matrix A
 *
 * https://stackoverflow.com/questions/26957900/should-the-dimension-of-work-array-on-xgehrd-and-xhseqr-routines-be-equal-on-eig
 * https://www.netlib.org/lapack/lapack-3.1.1/html/sgehrd.f.html
 * https://www.netlib.org/lapack/lapack-3.1.1/html/shseqr.f.html
 *
 * https://gensoft.pasteur.fr/docs/lapack/3.9.0/d8/ddc/group__real_g_ecomputational_ga971828f964b9d15b72ea12b3d8321d88.html
 *
 */
bool get_eigenvalues(vector<vector<double> >& a, vector<double>& eigen){
    if (a.size() < 2){
        fprintf(stderr, "error: invalid dimensions for A\n");
        return false;
    }
    if (a.size() != a[0].size()){
        fprintf(stderr, "error: A not symmetric\n");
        return false;
    }

    int N = a.size();
    int ILO = N; // Should this be 1 instead?
    int IHI = N; 
    int INFO;
    int LDA = N;
    
    // Create A array in LAPACK form
    float* a_lapack = (float*) malloc(sizeof(float)*N*N);
    int k = 0;
    for (int j = 0; j < N; ++j){
        for (int i = 0; i < N; ++i){
            a_lapack[i+j*N] = a[i][j];
        }
    }

    float* tau = (float*) malloc(sizeof(float)*(N-1)*(N-1));
    int LWORK = -1;
    float wkopt;
    
    // Query for optimal work value
    LAPACK_sgehrd(&N, &ILO, &IHI, a_lapack, &LDA, tau, &wkopt, &LWORK, &INFO);
    if (INFO != 0){
        fprintf(stderr, "SGEHRD workspace query failed\n");
        fprintf(stderr, "%dth argument invalid\n", -INFO);
        return false;
    }

    LWORK = int(wkopt);
    float* WORK = (float*) malloc(sizeof(float)*LWORK);

    LAPACK_sgehrd(&N, &ILO, &IHI, a_lapack, &LDA, tau, WORK, &LWORK, &INFO);
    if (INFO != 0){
        fprintf(stderr, "SGEHRD operation failed\n");
        fprintf(stderr, "%dth argument invalid\n", -INFO);
        return false;
    }

    LWORK = -1;
    char JOB = 'E';
    char COMPZ = 'N';
    int LDH = N;
    
    float* wr = (float*) malloc(sizeof(float)*N);
    float* wi = (float*) malloc(sizeof(float)*N);
    float* Z = (float*) malloc(sizeof(float)*N*N);
    int LDZ = N;

    LAPACK_shseqr(&JOB, &COMPZ, &N, &ILO, &IHI, a_lapack, &LDH, wr, wi, Z, &LDZ, &wkopt, &LWORK, &INFO);
    if (INFO != 0){
        fprintf(stderr, "SHSEQR workspace query failed\n");
        
        return false;
    }  
    LWORK = (int)wkopt;
    WORK = (float*)realloc((void*)WORK, sizeof(float)*LWORK);

    LAPACK_shseqr(&JOB, &COMPZ, &N, &ILO, &IHI, a_lapack, &LDH, wr, wi, Z, &LDZ, &wkopt, &LWORK, &INFO);
    if (INFO != 0){
        fprintf(stderr, "SHSEQR operation failed\n");
        if (INFO < 0){
            fprintf(stderr, "%dth argument invalid\n", -INFO);
        }
        else{
            fprintf(stderr, "Failed to compute some eigenvalues\n");
        }
        return false;
    }

    // Pull out eigenvalues
    eigen.clear();
    for (int i = 0; i < N; ++i){
        eigen.push_back(wr[i]);
    }
    
    free(Z);
    free(WORK);
    free(wr);
    free(wi);
    free(tau);
    free(a_lapack);
     
    return true;
}

