#include "ml_dmlc_tvm_native_c_api.h"  // generated by javah
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <iostream>

#include "jni_helper_func.h"

JavaVM *_jvm;

JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_nativeLibInit
  (JNIEnv *env, jobject obj) {
  return env->GetJavaVM(&_jvm);
}

JNIEXPORT jstring JNICALL Java_ml_dmlc_tvm_LibInfo_tvmGetLastError(JNIEnv * env, jobject obj) {
  return env->NewStringUTF(TVMGetLastError());
}

// Function
JNIEXPORT jint JNICALL Java_ml_dmlc_tvm_LibInfo_tvmFuncListGlobalNames(
  JNIEnv *env, jobject obj, jobject jfuncNames) {
  int outSize;
  const char **outArray;

  int ret = TVMFuncListGlobalNames(&outSize, &outArray);
  if (ret) {
    return ret;
  }

  jclass arrayClass = env->FindClass("scala/collection/mutable/ArrayBuffer");
  jmethodID arrayAppend = env->GetMethodID(arrayClass,
    "$plus$eq", "(Ljava/lang/Object;)Lscala/collection/mutable/ArrayBuffer;");

  // fill names
  for (int i = 0; i < outSize; ++i) {
    jstring jname = env->NewStringUTF(outArray[i]);
    env->CallObjectMethod(jfuncNames, arrayAppend, jname);
    env->DeleteLocalRef(jname);
  }

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
  JNIEnv *env, jobject obj, jlong jhandle, jobjectArray jargs, jobject jretVal) {
  jclass tvmArgClass = env->FindClass("ml/dmlc/tvm/types/TVMValue");
  jfieldID tvmArgId = env->GetFieldID(tvmArgClass, "argTypeId", "I");

  int numArgs = static_cast<int>(env->GetArrayLength(jargs));

  TVMValue *argValues = new TVMValue[numArgs];
  int *typeCodes = new int[numArgs];

  for (int i = 0; i < numArgs; ++i) {
    jobject jarg = env->GetObjectArrayElement(jargs, i);
    int argId = static_cast<int>(env->GetIntField(jarg, tvmArgId));
    TVMValue value;
    switch (argId) {
      case kInt:
        value.v_int64 = static_cast<int64_t>(getTVMValueLongField(env, jarg));
        break;
      case kFloat:
        value.v_float64 = static_cast<double>(getTVMValueDoubleField(env, jarg));
        break;
      case kStr:
        value.v_str = env->GetStringUTFChars(getTVMValueStringField(env, jarg), 0);
        // TODO: env->ReleaseStringUTFChars(jvalue, value);
        break;
      default:
        // TODO
        LOG(FATAL) << "Do NOT know how to handle argId " << argId;
    }
    typeCodes[i] = argId;
    argValues[i] = value;
  }

  TVMValue retVal;
  int retTypeCode;
  int ret = TVMFuncCall(reinterpret_cast<TVMFunctionHandle>(jhandle),
                        argValues, typeCodes, numArgs, &retVal, &retTypeCode);

  delete[] typeCodes;
  delete[] argValues;
  env->DeleteLocalRef(tvmArgClass);

  // return TVMValue object to Java
  jclass refTVMValueCls = env->FindClass("ml/dmlc/tvm/Base$RefTVMValue");
  jfieldID refTVMValueFid = env->GetFieldID(refTVMValueCls, "value", "Lml/dmlc/tvm/types/TVMValue;");

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
    default:
      // TODO
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
