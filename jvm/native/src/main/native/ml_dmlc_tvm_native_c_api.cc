/*!
 *  Copyright (c) 2017 by Contributors
 * \file ml_dmlc_tvm_native_c_api.cc
 * \brief tvm4j jni source file
 */
#include "ml_dmlc_tvm_native_c_api.h"  // generated by javah
#include <dlfcn.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/c_runtime_api.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <thread>

#include "jni_helper_func.h"

JavaVM *_jvm;
void *_tvmHandle;
struct TVMFuncArgsThreadLocalEntry {
  std::vector<TVMValue> tvmFuncArgValues;
  std::vector<int> tvmFuncArgTypes;
  // for later release
  std::vector<std::pair<jstring, const char *> > tvmFuncArgPushedStrs;
};
typedef dmlc::ThreadLocalStore<TVMFuncArgsThreadLocalEntry> TVMFuncArgsThreadLocalStore;

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_nativeLibInit
  (JNIEnv *env, jobject obj, jstring jtvmLibFile) {
  if (_tvmHandle == NULL) {
    const char *tvmLibFile = env->GetStringUTFChars(jtvmLibFile, 0);
    _tvmHandle = dlopen(tvmLibFile, RTLD_LAZY | RTLD_GLOBAL);
    env->ReleaseStringUTFChars(jtvmLibFile, tvmLibFile);
    if (!_tvmHandle) {
      fprintf(stderr, "%s\n", dlerror());
      return 1;
    }
  }
  return env->GetJavaVM(&_jvm);
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_shutdown(JNIEnv *env, jobject obj) {
  if (_tvmHandle) {
    dlclose(_tvmHandle);
  }
  return 0;
}

JNIEXPORT jstring JNICALL Java_ml_dmlc_tvm_LibInfo_tvmGetLastError(JNIEnv * env, jobject obj) {
  return env->NewStringUTF(TVMGetLastError());
}

// Function
JNIEXPORT void JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncPushArgLong(
  JNIEnv *env, jobject obj, jlong arg) {
  TVMValue value;
  value.v_int64 = static_cast<int64_t>(arg);
  TVMFuncArgsThreadLocalEntry *e = TVMFuncArgsThreadLocalStore::Get();
  e->tvmFuncArgValues.push_back(value);
  e->tvmFuncArgTypes.push_back(kInt);
}

JNIEXPORT void JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncPushArgDouble(
  JNIEnv *env, jobject obj, jdouble arg) {
  TVMValue value;
  value.v_float64 = static_cast<double>(arg);
  TVMFuncArgsThreadLocalEntry *e = TVMFuncArgsThreadLocalStore::Get();
  e->tvmFuncArgValues.push_back(value);
  e->tvmFuncArgTypes.push_back(kFloat);
}

JNIEXPORT void JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncPushArgString(
  JNIEnv *env, jobject obj, jstring arg) {
  TVMValue value;
  jstring garg = reinterpret_cast<jstring>(env->NewGlobalRef(arg));
  value.v_str = env->GetStringUTFChars(garg, 0);
  TVMFuncArgsThreadLocalEntry *e = TVMFuncArgsThreadLocalStore::Get();
  e->tvmFuncArgValues.push_back(value);
  e->tvmFuncArgTypes.push_back(kStr);
  // release string args later
  e->tvmFuncArgPushedStrs.push_back(std::make_pair(garg, value.v_str));
}

JNIEXPORT void JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncPushArgHandle(
  JNIEnv *env, jobject obj, jlong arg, jint argType) {
  TVMValue value;
  value.v_handle = reinterpret_cast<void *>(arg);
  TVMFuncArgsThreadLocalEntry *e = TVMFuncArgsThreadLocalStore::Get();
  e->tvmFuncArgValues.push_back(value);
  e->tvmFuncArgTypes.push_back(static_cast<int>(argType));
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncListGlobalNames(
  JNIEnv *env, jobject obj, jobject jfuncNames) {
  int outSize;
  const char **outArray;

  int ret = TVMFuncListGlobalNames(&outSize, &outArray);
  if (ret) {
    return ret;
  }

  jclass arrayClass = env->FindClass("java/util/List");
  jmethodID arrayAppend = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");

  // fill names
  for (int i = 0; i < outSize; ++i) {
    jstring jname = env->NewStringUTF(outArray[i]);
    env->CallObjectMethod(jfuncNames, arrayAppend, jname);
    env->DeleteLocalRef(jname);
  }

  env->DeleteLocalRef(arrayClass);

  return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncFree(
  JNIEnv *env, jobject obj, jlong jhandle) {
  return TVMFuncFree(reinterpret_cast<TVMFunctionHandle>(jhandle));
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncGetGlobal(
  JNIEnv *env, jobject obj, jstring jname, jobject jhandle) {
  TVMFunctionHandle handle;
  const char *name = env->GetStringUTFChars(jname, 0);
  int ret = TVMFuncGetGlobal(name, &handle);
  env->ReleaseStringUTFChars(jname, name);
  setLongField(env, jhandle, reinterpret_cast<jlong>(handle));
  return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncCall(
  JNIEnv *env, jobject obj, jlong jhandle, jobject jretVal) {
  TVMFuncArgsThreadLocalEntry *e = TVMFuncArgsThreadLocalStore::Get();
  int numArgs = e->tvmFuncArgValues.size();

  TVMValue retVal;
  int retTypeCode;
  int ret = TVMFuncCall(reinterpret_cast<TVMFunctionHandle>(jhandle),
    &e->tvmFuncArgValues[0], &e->tvmFuncArgTypes[0], numArgs, &retVal, &retTypeCode);

  for (auto iter = e->tvmFuncArgPushedStrs.cbegin();
        iter != e->tvmFuncArgPushedStrs.cend(); iter++) {
    env->ReleaseStringUTFChars(iter->first, iter->second);
    env->DeleteGlobalRef(iter->first);
  }
  e->tvmFuncArgPushedStrs.clear();
  e->tvmFuncArgTypes.clear();
  e->tvmFuncArgValues.clear();

  // return TVMValue object to Java
  jclass refTVMValueCls = env->FindClass("ml/dmlc/tvm/Base$RefTVMValue");
  jfieldID refTVMValueFid
    = env->GetFieldID(refTVMValueCls, "value", "Lml/dmlc/tvm/types/TVMValue;");

  switch (retTypeCode) {
    case kInt:
      env->SetObjectField(jretVal, refTVMValueFid,
        newTVMValueLong(env, static_cast<jlong>(retVal.v_int64)));
      break;
    case kFloat:
      env->SetObjectField(jretVal, refTVMValueFid,
        newTVMValueDouble(env, static_cast<jdouble>(retVal.v_float64)));
      break;
    case kModuleHandle:
      env->SetObjectField(jretVal, refTVMValueFid,
        newTVMValueModuleHandle(env, reinterpret_cast<jlong>(retVal.v_handle)));
      break;
    case kNull:
      env->SetObjectField(jretVal, refTVMValueFid,
        newObject(env, "ml/dmlc/tvm/types/TVMValueNull"));
      break;
    default:
      LOG(FATAL) << "Do NOT know how to handle return type code " << retTypeCode;
  }

  env->DeleteLocalRef(refTVMValueCls);

  return ret;
}

// Module
JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmModFree(
  JNIEnv *env, jobject obj, jlong jhandle) {
  return TVMFuncFree(reinterpret_cast<TVMModuleHandle>(jhandle));
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmModImport(
  JNIEnv *env, jobject obj, jlong jmod, jlong jdep) {
  return TVMModImport(reinterpret_cast<TVMModuleHandle>(jmod),
                      reinterpret_cast<TVMModuleHandle>(jdep));
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmModGetFunction(
  JNIEnv *env, jobject obj, jlong jhandle, jstring jname, jint jimport, jobject jret) {
  TVMFunctionHandle retFunc;

  const char *name = env->GetStringUTFChars(jname, 0);
  int ret = TVMModGetFunction(reinterpret_cast<TVMFunctionHandle>(jhandle),
                              name,
                              reinterpret_cast<int>(jimport),
                              &retFunc);
  env->ReleaseStringUTFChars(jname, name);

  setLongField(env, jret, reinterpret_cast<jlong>(retFunc));

  return ret;
}

// NDArray
JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmArrayFree(
  JNIEnv *env, jobject obj, jlong jhandle) {
  return TVMArrayFree(reinterpret_cast<TVMArrayHandle>(jhandle));
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmArrayAlloc(
  JNIEnv *env, jobject obj, jlongArray jshape, jobject jdtype, jobject jctx, jobject jret) {
  TVMType dtype;
  fromJavaDType(env, jdtype, &dtype);

  TVMContext ctx;
  fromJavaContext(env, jctx, &ctx);

  int ndim = static_cast<int>(env->GetArrayLength(jshape));

  TVMArrayHandle out;

  jlong *shapeArray = env->GetLongArrayElements(jshape, NULL);
  int ret = TVMArrayAlloc(reinterpret_cast<const tvm_index_t*>(shapeArray), ndim, dtype, ctx, &out);
  env->ReleaseLongArrayElements(jshape, shapeArray, 0);

  setLongField(env, jret, reinterpret_cast<jlong>(out));

  return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmArrayGetShape(
  JNIEnv *env, jobject obj, jlong jhandle, jobject jshape) {
  TVMArray *array = reinterpret_cast<TVMArray *>(jhandle);
  int64_t *shape = array->shape;
  int ndim = array->ndim;

  // fill shape buffer
  jclass longClass = env->FindClass("java/lang/Long");
  jmethodID newLong = env->GetMethodID(longClass, "<init>", "(J)V");

  jclass arrayClass = env->FindClass("java/util/List");
  jmethodID arrayAppend = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");
  for (int i = 0; i < ndim; ++i) {
    jobject data = env->NewObject(longClass, newLong, static_cast<jlong>(shape[i]));
    env->CallObjectMethod(jshape, arrayAppend, data);
    env->DeleteLocalRef(data);
  }
  env->DeleteLocalRef(longClass);
  env->DeleteLocalRef(arrayClass);

  return 0;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmArrayCopyFromTo(
  JNIEnv *env, jobject obj, jlong jfrom, jlong jto) {
  return TVMArrayCopyFromTo(reinterpret_cast<TVMArrayHandle>(jfrom),
                            reinterpret_cast<TVMArrayHandle>(jto), NULL);
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmArrayCopyFromJArray(
  JNIEnv *env, jobject obj, jbyteArray jarr, jlong jfrom, jlong jto) {
  jbyte *data = env->GetByteArrayElements(jarr, NULL);

  TVMArray *from = reinterpret_cast<TVMArray *>(jfrom);
  from->data = static_cast<void *>(data);

  int ret = TVMArrayCopyFromTo(static_cast<TVMArrayHandle>(from),
                               reinterpret_cast<TVMArrayHandle>(jto), NULL);

  from->data = NULL;
  env->ReleaseByteArrayElements(jarr, data, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmArrayCopyToJArray(
  JNIEnv *env, jobject obj, jlong jfrom, jbyteArray jarr) {
  TVMArray *from = reinterpret_cast<TVMArray *>(jfrom);
  int size = static_cast<int>(env->GetArrayLength(jarr));
  jbyte *pdata = env->GetByteArrayElements(jarr, NULL);
  int ret = 0;
  if (memcpy(static_cast<void *>(pdata), from->data, size) == NULL) {
    ret = 1;
  }
  env->ReleaseByteArrayElements(jarr, pdata, 0);  // copy back to java array automatically
  return ret;
}

// Context
JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmSynchronize(
  JNIEnv *env, jobject obj, jobject jctx) {
  TVMContext ctx;
  fromJavaContext(env, jctx, &ctx);
  return TVMSynchronize(ctx, NULL);
}
