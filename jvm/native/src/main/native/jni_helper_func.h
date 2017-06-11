/*!
 *  Copyright (c) 2017 by Contributors
 * \file jni_helper_func.h
 * \brief Helper functions for operating JVM objects
 */
#include <jni.h>

#ifndef TVM_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_
#define TVM_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_

// Helper functions for TVMValue
jlong getTVMValueLongField(JNIEnv *env, jobject obj) {
  jclass cls = env->FindClass("ml/dmlc/tvm/types/TVMValueLong");
  jfieldID fid = env->GetFieldID(cls, "value", "J");
  jlong ret = env->GetLongField(obj, fid);
  env->DeleteLocalRef(cls);
  return ret;
}

jdouble getTVMValueDoubleField(JNIEnv *env, jobject obj) {
  jclass cls = env->FindClass("ml/dmlc/tvm/types/TVMValueDouble");
  jfieldID fid = env->GetFieldID(cls, "value", "D");
  jdouble ret = env->GetDoubleField(obj, fid);
  env->DeleteLocalRef(cls);
  return ret;
}

jobject newTVMValueLong(JNIEnv *env, jlong value) {
  jclass cls = env->FindClass("ml/dmlc/tvm/types/TVMValueLong");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMValueDouble(JNIEnv *env, jdouble value) {
  jclass cls = env->FindClass("ml/dmlc/tvm/types/TVMValueDouble");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(D)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

#endif // TVM_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_
