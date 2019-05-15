/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file jni_helper_func.h
 * \brief Helper functions for operating JVM objects
 */
#include <jni.h>

#ifndef TVM4J_JNI_MAIN_NATIVE_JNI_HELPER_FUNC_H_
#define TVM4J_JNI_MAIN_NATIVE_JNI_HELPER_FUNC_H_

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
jlong getTVMValueLongField(JNIEnv *env, jobject obj,
  const char *clsname = "ml/dmlc/tvm/TVMValueLong") {
  jclass cls = env->FindClass(clsname);
  jfieldID fid = env->GetFieldID(cls, "value", "J");
  jlong ret = env->GetLongField(obj, fid);
  env->DeleteLocalRef(cls);
  return ret;
}

jdouble getTVMValueDoubleField(JNIEnv *env, jobject obj) {
  jclass cls = env->FindClass("ml/dmlc/tvm/TVMValueDouble");
  jfieldID fid = env->GetFieldID(cls, "value", "D");
  jdouble ret = env->GetDoubleField(obj, fid);
  env->DeleteLocalRef(cls);
  return ret;
}

jstring getTVMValueStringField(JNIEnv *env, jobject obj) {
  jclass cls = env->FindClass("ml/dmlc/tvm/TVMValueString");
  jfieldID fid = env->GetFieldID(cls, "value", "Ljava/lang/String;");
  jstring ret = static_cast<jstring>(env->GetObjectField(obj, fid));
  env->DeleteLocalRef(cls);
  return ret;
}

jobject newTVMValueHandle(JNIEnv *env, jlong value) {
  jclass cls = env->FindClass("ml/dmlc/tvm/TVMValueHandle");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMValueLong(JNIEnv *env, jlong value) {
  jclass cls = env->FindClass("ml/dmlc/tvm/TVMValueLong");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMValueDouble(JNIEnv *env, jdouble value) {
  jclass cls = env->FindClass("ml/dmlc/tvm/TVMValueDouble");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(D)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newTVMValueString(JNIEnv *env, const char *value) {
  jstring jvalue = env->NewStringUTF(value);
  jclass cls = env->FindClass("ml/dmlc/tvm/TVMValueString");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
  jobject object = env->NewObject(cls, constructor, jvalue);
  env->DeleteLocalRef(cls);
  env->DeleteLocalRef(jvalue);
  return object;
}

jobject newTVMValueBytes(JNIEnv *env, const TVMByteArray *arr) {
  jbyteArray jarr = env->NewByteArray(arr->size);
  env->SetByteArrayRegion(jarr, 0, arr->size,
      reinterpret_cast<jbyte *>(const_cast<char *>(arr->data)));
  jclass cls = env->FindClass("ml/dmlc/tvm/TVMValueBytes");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "([B)V");
  jobject object = env->NewObject(cls, constructor, jarr);
  env->DeleteLocalRef(cls);
  env->DeleteLocalRef(jarr);
  return object;
}

jobject newModule(JNIEnv *env, jlong value) {
  jclass cls = env->FindClass("ml/dmlc/tvm/Module");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newFunction(JNIEnv *env, jlong value) {
  jclass cls = env->FindClass("ml/dmlc/tvm/Function");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
  jobject object = env->NewObject(cls, constructor, value);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newNDArray(JNIEnv *env, jlong handle, jboolean isview) {
  jclass cls = env->FindClass("ml/dmlc/tvm/NDArrayBase");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(JZ)V");
  jobject object = env->NewObject(cls, constructor, handle, isview);
  env->DeleteLocalRef(cls);
  return object;
}

jobject newObject(JNIEnv *env, const char *clsname) {
  jclass cls = env->FindClass(clsname);
  jmethodID constructor = env->GetMethodID(cls, "<init>", "()V");
  jobject object = env->NewObject(cls, constructor);
  env->DeleteLocalRef(cls);
  return object;
}

void fromJavaDType(JNIEnv *env, jobject jdtype, TVMType *dtype) {
  jclass tvmTypeClass = env->FindClass("ml/dmlc/tvm/TVMType");
  dtype->code = (uint8_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "typeCode", "I")));
  dtype->bits = (uint8_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "bits", "I")));
  dtype->lanes = (uint16_t)(env->GetIntField(jdtype, env->GetFieldID(tvmTypeClass, "lanes", "I")));
  env->DeleteLocalRef(tvmTypeClass);
}

void fromJavaContext(JNIEnv *env, jobject jctx, TVMContext *ctx) {
  jclass tvmContextClass = env->FindClass("ml/dmlc/tvm/TVMContext");
  ctx->device_type = static_cast<DLDeviceType>(env->GetIntField(jctx,
    env->GetFieldID(tvmContextClass, "deviceType", "I")));
  ctx->device_id = static_cast<int>(env->GetIntField(jctx,
    env->GetFieldID(tvmContextClass, "deviceId", "I")));
  env->DeleteLocalRef(tvmContextClass);
}

jobject tvmRetValueToJava(JNIEnv *env, TVMValue value, int tcode) {
  switch (tcode) {
    case kDLUInt:
    case kDLInt:
      return newTVMValueLong(env, static_cast<jlong>(value.v_int64));
    case kDLFloat:
      return newTVMValueDouble(env, static_cast<jdouble>(value.v_float64));
    case kHandle:
      return newTVMValueHandle(env, reinterpret_cast<jlong>(value.v_handle));
    case kModuleHandle:
      return newModule(env, reinterpret_cast<jlong>(value.v_handle));
    case kFuncHandle:
      return newFunction(env, reinterpret_cast<jlong>(value.v_handle));
    case kArrayHandle:
      return newNDArray(env, reinterpret_cast<jlong>(value.v_handle), true);
    case kNDArrayContainer:
      return newNDArray(env, reinterpret_cast<jlong>(value.v_handle), false);
    case kStr:
      return newTVMValueString(env, value.v_str);
    case kBytes:
      return newTVMValueBytes(env, reinterpret_cast<TVMByteArray *>(value.v_handle));
    case kNull:
      return newObject(env, "ml/dmlc/tvm/TVMValueNull");
    default:
      LOG(FATAL) << "Do NOT know how to handle return type code " << tcode;
  }
  return NULL;
}

#endif  // TVM4J_JNI_MAIN_NATIVE_JNI_HELPER_FUNC_H_
