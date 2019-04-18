#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>

#include "../CSC.h"
#include "../CSR.h"
#include "../IO.h"
#include "../utility.h"

using namespace std;

extern "C" {
#include "../GTgraph/R-MAT/defs.h"
#include "../GTgraph/R-MAT/graph.h"
#include "../GTgraph/R-MAT/init.h"
}

#ifndef _SAMPLE_COMMON_HPP_
#define _SAMPLE_COMMON_HPP_

#ifdef CPP
#define MALLOC "new"
#elif defined IMM
#define MALLOC "mm"
#elif defined TBB
#define MALLOC "tbb"
#else
#define MALLOC "tbb"
#endif


enum generator_type {
  rmat_graph,
  er_graph,
};

template <class INDEXTYPE, class VALUETYPE>
void SetInputMatricesAsCSC(CSC<INDEXTYPE, VALUETYPE> &A_csc,
                           CSC<INDEXTYPE, VALUETYPE> &B_csc, char **argv) {
  bool binary = false;
  bool gen = false;
  string inputname1, inputname2, outputname;
  int edgefactor, scale, r_scale;
  generator_type gtype;

  A_csc.make_empty();
  B_csc.make_empty();

  if (string(argv[1]) == string("gen")) {
    gen = true;
    cout << "Using synthetically generated data of scale " << argv[3]
         << " and edgefactor " << argv[4] << endl;
    scale = atoi(argv[3]);
    edgefactor = atoi(argv[4]);
    r_scale = scale;
    if (string(argv[2]) == string("rmat")) {
      cout << "RMAT Generator" << endl << endl;
      gtype = rmat_graph;
    } else {
      // Erdos-Renyi graph
      cout << "ER Generator" << endl << endl;
      gtype = er_graph;
    }
  } else {
    inputname1 = argv[2];
    inputname2 = argv[3];
    string isbinary(argv[1]);
    if (isbinary == "text") {
      binary = false;
    } else if (isbinary == "binary") {
      binary = true;
    } else {
      cout << "unrecognized option, assuming text file" << endl;
    }
  }

  if (gen) {
    double a, b, c, d;
    if (gtype == rmat_graph) {
      a = 0.57;
      b = 0.19;
      c = 0.19;
      d = 0.05;
    } else {
      a = b = c = d = 0.25;
    }
    getParams();
    setGTgraphParams(scale, edgefactor, a, b, c, d);
    graph G1;
    graphGen(&G1);
    cerr << "Generator returned" << endl;
    A_csc = *(new CSC<INDEXTYPE, VALUETYPE>(G1)); // convert to CSC
    if (STORE_IN_MEMORY) {
      free(G1.start);
      free(G1.end);
      free(G1.w);
    }

    graph G2;
    graphGen(&G2);
    cerr << "Generator returned" << endl; // convert to CSC
    B_csc = *(new CSC<INDEXTYPE, VALUETYPE>(G2));

    if (STORE_IN_MEMORY) {
      free(G2.start);
      free(G2.end);
      free(G2.w);
    }
  } else {
    if (binary) {
      ReadBinary(inputname1, A_csc);
      ReadBinary(inputname2, B_csc);
    } else {
      cout << "reading input matrices in text (ascii)... " << endl;
      ReadASCII(inputname1, A_csc);
      ReadASCII(inputname2, B_csc);
      stringstream ss1(inputname1);
      string cur;
      vector<string> v1;
      while (getline(ss1, cur, '.')) {
        v1.push_back(cur);
      }

      stringstream ss2(v1[v1.size() - 2]);
      vector<string> v2;
      while (getline(ss2, cur, '/')) {
        v2.push_back(cur);
      }
      inputname1 = v2[v2.size() - 1];
    }
  }
}

template <class INDEXTYPE, class VALUETYPE>
void SetInputMatricesAsCSR(CSR<INDEXTYPE, VALUETYPE> &A_csr,
                           CSR<INDEXTYPE, VALUETYPE> &B_csr, char **argv) {
  CSC<INDEXTYPE, VALUETYPE> A_csc, B_csc;

  A_csr.make_empty();
  B_csr.make_empty();

  SetInputMatricesAsCSC(A_csc, B_csc, argv);

  A_csr = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc));
  B_csr = *(new CSR<INDEXTYPE, VALUETYPE>(B_csc));
}

#endif
