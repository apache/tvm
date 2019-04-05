/*!
 *  Copyright (c) 2017 by Contributors
 * \file ml_dmlc_tvm_native_c_api.cc
 * \brief tvm4j jni source file
 */
#include "ml_dmlc_tvm_native_c_api.h"  // generated by javah
#ifdef TVM4J_ANDROID
#include "tvm_runtime.h"
#else
#include <dlfcn.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/c_runtime_api.h>
#endif
#include <iostream>
#include <cstring>
#include <vector>
#include <thread>

#include "jni_helper_func.h"

JavaVM *_jvm;
void *_tvmHandle = nullptr;
struct TVMFuncArgsThreadLocalEntry {
  std::vector<TVMValue> tvmFuncArgValues;
  std::vector<int> tvmFuncArgTypes;
  // for later release
  std::vector<std::pair<jstring, const char *> > tvmFuncArgPushedStrs;
  std::vector<std::pair<jbyteArray, TVMByteArray *> > tvmFuncArgPushedBytes;
};
typedef dmlc::ThreadLocalStore<TVMFuncArgsThreadLocalEntry> TVMFuncArgsThreadLocalStore;

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_nativeLibInit
  (JNIEnv *env, jobject obj, jstring jtvmLibFile) {
  if (_tvmHandle == NULL && !env->IsSameObject(jtvmLibFile, NULL)) {
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
  e->tvmFuncArgTypes.push_back(kDLInt);
}

JNIEXPORT void JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncPushArgDouble(
  JNIEnv *env, jobject obj, jdouble arg) {
  TVMValue value;
  value.v_float64 = static_cast<double>(arg);
  TVMFuncArgsThreadLocalEntry *e = TVMFuncArgsThreadLocalStore::Get();
  e->tvmFuncArgValues.push_back(value);
  e->tvmFuncArgTypes.push_back(kDLFloat);
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

JNIEXPORT void JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncPushArgBytes(
  JNIEnv *env, jobject obj, jbyteArray arg) {
  jbyteArray garg = reinterpret_cast<jbyteArray>(env->NewGlobalRef(arg));
  jbyte *data = env->GetByteArrayElements(garg, 0);

  TVMByteArray *byteArray = new TVMByteArray();
  byteArray->size = static_cast<size_t>(env->GetArrayLength(garg));
  byteArray->data = reinterpret_cast<const char *>(data);

  TVMValue value;
  value.v_handle = reinterpret_cast<void *>(byteArray);

  TVMFuncArgsThreadLocalEntry *e = TVMFuncArgsThreadLocalStore::Get();
  e->tvmFuncArgValues.push_back(value);
  e->tvmFuncArgTypes.push_back(kBytes);

  e->tvmFuncArgPushedBytes.push_back(std::make_pair(garg, byteArray));
  // release (garg, data), byteArray later
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
    env->CallBooleanMethod(jfuncNames, arrayAppend, jname);
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

  // function can be invoked recursively,
  // thus we copy the pushed arguments here.
  auto argValues = e->tvmFuncArgValues;
  auto argTypes = e->tvmFuncArgTypes;
  auto pushedStrs = e->tvmFuncArgPushedStrs;
  auto pushedBytes = e->tvmFuncArgPushedBytes;

  e->tvmFuncArgPushedStrs.clear();
  e->tvmFuncArgPushedBytes.clear();
  e->tvmFuncArgTypes.clear();
  e->tvmFuncArgValues.clear();

  int ret = TVMFuncCall(reinterpret_cast<TVMFunctionHandle>(jhandle),
    &argValues[0], &argTypes[0], numArgs, &retVal, &retTypeCode);

  if (ret != 0) {
    return ret;
  }

  for (auto iter = pushedStrs.cbegin(); iter != pushedStrs.cend(); iter++) {
    env->ReleaseStringUTFChars(iter->first, iter->second);
    env->DeleteGlobalRef(iter->first);
  }
  for (auto iter = pushedBytes.cbegin(); iter != pushedBytes.cend(); iter++) {
    env->ReleaseByteArrayElements(iter->first,
        reinterpret_cast<jbyte *>(const_cast<char *>(iter->second->data)), 0);
    env->DeleteGlobalRef(iter->first);
    delete iter->second;
  }

  // return TVMValue object to Java
  jclass refTVMValueCls = env->FindClass("ml/dmlc/tvm/Base$RefTVMValue");
  jfieldID refTVMValueFid
    = env->GetFieldID(refTVMValueCls, "value", "Lml/dmlc/tvm/TVMValue;");

  env->SetObjectField(jretVal, refTVMValueFid, tvmRetValueToJava(env, retVal, retTypeCode));

  env->DeleteLocalRef(refTVMValueCls);

  return ret;
}

// Callback function
extern "C" int funcInvokeCallback(TVMValue *args,
    int *typeCodes, int numArgs, TVMRetValueHandle ret, void *resourceHandle) {
  JNIEnv *env;
  int jniStatus = _jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
  if (jniStatus == JNI_EDETACHED) {
  #ifdef TVM4J_ANDROID
    _jvm->AttachCurrentThread(&env, nullptr);
  #else
    _jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), nullptr);
  #endif
  } else {
    CHECK(jniStatus == JNI_OK);
  }

  jclass tvmValueCls = env->FindClass("ml/dmlc/tvm/TVMValue");
  jobjectArray jargs = env->NewObjectArray(numArgs, tvmValueCls, 0);
  for (int i = 0; i < numArgs; ++i) {
    TVMValue arg = args[i];
    int tcode = typeCodes[i];
    if (tcode == kNodeHandle || tcode == kFuncHandle || tcode == kModuleHandle) {
      TVMCbArgToReturn(&arg, tcode);
    }
    jobject jarg = tvmRetValueToJava(env, arg, tcode);
    env->SetObjectArrayElement(jargs, i, jarg);
  }

  jclass clsFunc = env->FindClass("ml/dmlc/tvm/Function");
  jmethodID invokeRegisteredCbFunc = env->GetStaticMethodID(clsFunc, "invokeRegisteredCbFunc",
      "(Lml/dmlc/tvm/Function$Callback;[Lml/dmlc/tvm/TVMValue;)Ljava/lang/Object;");
  jmethodID pushArgToStack = env->GetStaticMethodID(clsFunc, "pushArgToStack",
      "(Ljava/lang/Object;)V");

  jobject jretValue = env->CallStaticObjectMethod(clsFunc, invokeRegisteredCbFunc,
      reinterpret_cast<jobject>(resourceHandle), jargs);

  TVMFuncArgsThreadLocalEntry *e = TVMFuncArgsThreadLocalStore::Get();
  const int prevNumStrArg = e->tvmFuncArgPushedStrs.size();
  const int prevNumBytesArg = e->tvmFuncArgPushedBytes.size();

  // convert returned (java) TVMValue to (C) TVMValue
  env->CallStaticVoidMethod(clsFunc, pushArgToStack, jretValue);

  TVMValue retValue = e->tvmFuncArgValues.back();
  e->tvmFuncArgValues.pop_back();

  int retCode = e->tvmFuncArgTypes.back();
  e->tvmFuncArgTypes.pop_back();

  // set back the return value
  TVMCFuncSetReturn(ret, &retValue, &retCode, 1);

  // release allocated strings.
  if (e->tvmFuncArgPushedStrs.size() > prevNumStrArg) {
    const auto &pairArg = e->tvmFuncArgPushedStrs.back();
    env->ReleaseStringUTFChars(pairArg.first, pairArg.second);
    env->DeleteGlobalRef(pairArg.first);
    e->tvmFuncArgPushedStrs.pop_back();
  }
  // release allocated bytes.
  if (e->tvmFuncArgPushedBytes.size() > prevNumBytesArg) {
    const auto &pairArg = e->tvmFuncArgPushedBytes.back();
    env->ReleaseByteArrayElements(pairArg.first,
        reinterpret_cast<jbyte *>(const_cast<char *>(pairArg.second->data)), 0);
    env->DeleteGlobalRef(pairArg.first);
    delete pairArg.second;
    e->tvmFuncArgPushedBytes.pop_back();
  }

  env->DeleteLocalRef(clsFunc);
  env->DeleteLocalRef(tvmValueCls);

  return 0;
}

// Free callback function
extern "C" void funcFreeCallback(void *resourceHandle) {
  JNIEnv *env;
  int jniStatus = _jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
  if (jniStatus == JNI_EDETACHED) {
  #ifdef TVM4J_ANDROID
    _jvm->AttachCurrentThread(&env, nullptr);
  #else
    _jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), nullptr);
  #endif
  } else {
    CHECK(jniStatus == JNI_OK);
  }
  env->DeleteGlobalRef(reinterpret_cast<jobject>(resourceHandle));
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncCreateFromCFunc(
  JNIEnv *env, jobject obj, jobject jfunction, jobject jretHandle) {
  TVMFunctionHandle out;
  int ret = TVMFuncCreateFromCFunc(reinterpret_cast<TVMPackedCFunc>(&funcInvokeCallback),
                                   reinterpret_cast<void *>(env->NewGlobalRef(jfunction)),
                                   reinterpret_cast<TVMPackedCFuncFinalizer>(&funcFreeCallback),
                                   &out);
  setLongField(env, jretHandle, reinterpret_cast<jlong>(out));
  return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncRegisterGlobal(
  JNIEnv *env, jobject obj, jstring jname, jlong jhandle, jint joverride) {
  const char *name = env->GetStringUTFChars(jname, 0);
  int ret = TVMFuncRegisterGlobal(
      name, reinterpret_cast<TVMFunctionHandle>(jhandle), reinterpret_cast<int>(joverride));
  env->ReleaseStringUTFChars(jname, name);
  return ret;
}

// Module
JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmModFree(
  JNIEnv *env, jobject obj, jlong jhandle) {
  return TVMModFree(reinterpret_cast<TVMModuleHandle>(jhandle));
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
  JNIEnv *env, jobject obj, jlongArray jshape, jint jdtypeCode,
  jint jdtypeBits, jint jdtypeLanes, jint jdeviceType, jint jdeviceId, jobject jret) {
  int ndim = static_cast<int>(env->GetArrayLength(jshape));

  TVMArrayHandle out;

  jlong *shapeArray = env->GetLongArrayElements(jshape, NULL);
  int ret = TVMArrayAlloc(
      reinterpret_cast<const tvm_index_t*>(shapeArray),
      ndim,
      static_cast<int>(jdtypeCode),
      static_cast<int>(jdtypeBits),
      static_cast<int>(jdtypeLanes),
      static_cast<int>(jdeviceType),
      static_cast<int>(jdeviceId),
      &out);
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
    env->CallBooleanMethod(jshape, arrayAppend, data);
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
  JNIEnv *env, jint deviceType, jint deviceId) {
  return TVMSynchronize(static_cast<int>(deviceType), static_cast<int>(deviceId), NULL);
}
