/*!
 *  Copyright (c) 2017 by Contributors
 * \file jni_helper_func.h
 * \brief Helper functions for operating JVM objects
 */
#include <jni.h>

#ifndef TVM_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_
#define TVM_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_

// Helper functions for RefXXX getter & setter
jlong getLongField(JNIEnv *env, jobject obj) {
  jclass refClass = env->FindClass("ml/dmlc/tvm/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  jlong ret = env->GetLongField(obj, refFid);
  env->DeleteLocalRef(refClass);
  return ret;
}

jint getIntField(JNIEnv *env, jobject obj) {
  jclass refClass = env->FindClass("ml/dmlc/tvm/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  jint ret = env->GetIntField(obj, refFid);
  env->DeleteLocalRef(refClass);
  return ret;
}

void setIntField(JNIEnv *env, jobject obj, jint value) {
  jclass refClass = env->FindClass("ml/dmlc/tvm/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  env->SetIntField(obj, refFid, value);
  env->DeleteLocalRef(refClass);
}

void setLongField(JNIEnv *env, jobject obj, jlong value) {
  jclass refClass = env->FindClass("ml/dmlc/tvm/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  env->SetLongField(obj, refFid, value);
  env->DeleteLocalRef(refClass);
}

void setStringField(JNIEnv *env, jobject obj, const char *value) {
  jclass refClass = env->FindClass("ml/dmlc/tvm/Base$RefString");
  jfieldID refFid = env->GetFieldID(refClass, "value", "Ljava/lang/String;");
  env->SetObjectField(obj, refFid, env->NewStringUTF(value));
  env->DeleteLocalRef(refClass);
}

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

jstring getTVMValueStringField(JNIEnv *env, jobject obj) {
  jclass cls = env->FindClass("ml/dmlc/tvm/types/TVMValueString");
  jfieldID fid = env->GetFieldID(cls, "value", "Ljava/lang/String;");
  jstring ret = static_cast<jstring>(env->GetObjectField(obj, fid));
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

jobject newTVMValueModuleHandle(JNIEnv *env, jlong value) {
  jclass cls = env->FindClass("ml/dmlc/tvm/types/TVMValueModuleHandle");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

void fromJavaDType(JNIEnv *env, jobject jdtype, TVMType &dtype) {
  jclass tvmTypeClass = env->FindClass("ml/dmlc/tvm/types/TVMType");
  dtype.code = (uint8_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "typeCode", "I")));
  dtype.bits = (uint8_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "bits", "I")));
  dtype.lanes = (uint16_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "lanes", "I")));
  env->DeleteLocalRef(tvmTypeClass);
}

void fromJavaContext(JNIEnv *env, jobject jctx, TVMContext &ctx) {
  jclass tvmContextClass = env->FindClass("ml/dmlc/tvm/types/TVMContext");
  ctx.device_type = static_cast<DLDeviceType>(env->GetIntField(jctx,
    env->GetFieldID(tvmContextClass, "deviceType", "I")));
  ctx.device_id = static_cast<int>(env->GetIntField(jctx,
    env->GetFieldID(tvmContextClass, "deviceId", "I")));
  env->DeleteLocalRef(tvmContextClass);
}

#endif // TVM_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_
