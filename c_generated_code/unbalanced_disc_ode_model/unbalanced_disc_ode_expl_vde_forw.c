/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) unbalanced_disc_ode_expl_vde_forw_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s1[9] = {2, 2, 0, 2, 4, 0, 1, 0, 1};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[3] = {0, 0, 0};

/* unbalanced_disc_ode_expl_vde_forw:(i0[2],i1[2x2],i2[2],i3,i4[])->(o0[2],o1[2x2],o2[2]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7;
  a0=-1.2560237022669189e+02;
  a1=arg[0]? arg[0][1] : 0;
  a2=sin(a1);
  a2=(a0*a2);
  a3=2.5127323476804801e+00;
  a4=arg[0]? arg[0][0] : 0;
  a5=(a3*a4);
  a2=(a2-a5);
  a5=2.6404248175282628e+01;
  a6=arg[3]? arg[3][0] : 0;
  a6=(a5*a6);
  a2=(a2+a6);
  if (res[0]!=0) res[0][0]=a2;
  if (res[0]!=0) res[0][1]=a4;
  a4=cos(a1);
  a2=arg[1]? arg[1][1] : 0;
  a2=(a4*a2);
  a2=(a0*a2);
  a6=arg[1]? arg[1][0] : 0;
  a7=(a3*a6);
  a2=(a2-a7);
  if (res[1]!=0) res[1][0]=a2;
  if (res[1]!=0) res[1][1]=a6;
  a6=arg[1]? arg[1][3] : 0;
  a4=(a4*a6);
  a4=(a0*a4);
  a6=arg[1]? arg[1][2] : 0;
  a2=(a3*a6);
  a4=(a4-a2);
  if (res[1]!=0) res[1][2]=a4;
  if (res[1]!=0) res[1][3]=a6;
  a1=cos(a1);
  a6=arg[2]? arg[2][1] : 0;
  a1=(a1*a6);
  a0=(a0*a1);
  a1=arg[2]? arg[2][0] : 0;
  a3=(a3*a1);
  a0=(a0-a3);
  a5=(a5+a0);
  if (res[2]!=0) res[2][0]=a5;
  if (res[2]!=0) res[2][1]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int unbalanced_disc_ode_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int unbalanced_disc_ode_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int unbalanced_disc_ode_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void unbalanced_disc_ode_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int unbalanced_disc_ode_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void unbalanced_disc_ode_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void unbalanced_disc_ode_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void unbalanced_disc_ode_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int unbalanced_disc_ode_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int unbalanced_disc_ode_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real unbalanced_disc_ode_expl_vde_forw_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* unbalanced_disc_ode_expl_vde_forw_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* unbalanced_disc_ode_expl_vde_forw_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* unbalanced_disc_ode_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    case 3: return casadi_s2;
    case 4: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* unbalanced_disc_ode_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int unbalanced_disc_ode_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
