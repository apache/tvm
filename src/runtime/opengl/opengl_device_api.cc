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
 *  Copyright (c) 2017 by Contributors
 * \file opengl_device_api.cc
 */
#include <tvm/runtime/registry.h>
#include <cstring>
#include "opengl_common.h"
#include "opengl_module.h"

namespace tvm {
namespace runtime {
namespace gl {

/*!
 * \brief Turn OpenGL error enum to string.
 */
static const char* GLGetErrorString(GLenum error) {
  switch (error) {
    case GL_NO_ERROR:
      return "GL_NO_ERROR";
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
#if !defined(__APPLE__)
    case GL_STACK_OVERFLOW:
      return "GL_STACK_OVERFLOW";
    case GL_STACK_UNDERFLOW:
      return "GL_STACK_UNDERFLOW";
#endif
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
    default:
      return "Unknown OpenGL error code";
  }
}

/*!
 * \brief Get the latest error.
 */
void OpenGLWorkspace::CheckOpenGLError() {
  GLenum err = gl->GetError();
  CHECK_EQ(err, GL_NO_ERROR) << "OpenGL error, code=" << err << ": "
                             << gl::GLGetErrorString(err);
}

/*!
 * \brief Protected OpenGL call.
 * \param func Expression to call.
 */
#define OPENGL_CALL(func)                                                      \
  {                                                                            \
    (func);                                                                    \
    CheckOpenGLError();                                                        \
  }

/*!
 * \brief The error handling callback passed to GLFW.
 */
void GlfwErrorCallback(int err, const char* str) {
  LOG(FATAL) << "Error: [" << err << "] " << str;
}

const std::shared_ptr<OpenGLWorkspace>& OpenGLWorkspace::Global() {
  static std::shared_ptr<OpenGLWorkspace> inst(new OpenGLWorkspace);
  return inst;
}

void OpenGLWorkspace::SetDevice(TVMContext ctx) {
  CHECK_EQ(ctx.device_type, static_cast<int>(kOpenGL))
    << "Device type must be OpenGL.";
  CHECK_EQ(ctx.device_id, 0) << "Only support 1 OpenGL \"device\".";
}

void OpenGLWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  switch (kind) {
    case kExist: {
      *rv = static_cast<int>(ctx.device_id == 0);
      break;
    }
    case kMaxThreadsPerBlock: {
      GLint max_texture_size;
      OPENGL_CALL(gl->GetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size));
      break;
    }
    case kWarpSize: {
      *rv = 1;
      break;
    }
    case kMaxSharedMemoryPerBlock: return;
    case kComputeVersion: {
      break;
    }
    case kDeviceName: return;
    case kMaxClockRate: return;
    case kMultiProcessorCount: return;
    case kMaxThreadDimensions: return;
  }
}

void* OpenGLWorkspace::AllocDataSpace(
    TVMContext ctx, size_t nbytes, size_t alignment, TVMType type_hint) {
  return reinterpret_cast<void*>(new Texture(CreateTexture(type_hint, nbytes)));
}

void OpenGLWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  delete reinterpret_cast<Texture*>(ptr);
}

void OpenGLWorkspace::CopyDataFromTo(const void* from,
                                     size_t from_offset,
                                     void* to,
                                     size_t to_offset,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMType type_hint,
                                     TVMStreamHandle stream) {
  CHECK(stream == nullptr);

  // TODO(zhixunt): This is a nasty hack to avoid comparison between
  // incompatible enums. We should add kOpenGL to dlpack.
  constexpr int gl_devtype = kOpenGL;
  std::tuple<int, int> type_from_to(ctx_from.device_type, ctx_to.device_type);

  if (type_from_to == std::make_tuple(gl_devtype, gl_devtype)) {
    auto from_texture = static_cast<const Texture*>(from);
    auto to_texture = static_cast<Texture*>(to);
    auto temp_buffer = std::unique_ptr<char[]>(new char[size]);
    CHECK(from_texture->format_ == to_texture->format_);
    auto elemsz = from_texture->elemsz();
    auto from_begin = static_cast<GLint>(from_offset / elemsz);
    auto to_begin = static_cast<GLint>(to_offset / elemsz);
    auto nelems = static_cast<GLsizei>(size / elemsz);
    GetTextureData(from_texture, from_begin, nelems, temp_buffer.get());
    PutTextureData(to_texture, to_begin, nelems, temp_buffer.get());

  } else if (type_from_to == std::make_tuple(gl_devtype, kDLCPU)) {
    auto texture = static_cast<const Texture*>(from);
    void *data = static_cast<char *>(to) + to_offset;
    auto elemsz = texture->elemsz();
    auto begin = static_cast<GLint>(from_offset / elemsz);
    auto nelems = static_cast<GLsizei>(size / elemsz);
    GetTextureData(texture, begin, nelems, data);

  } else if (type_from_to == std::make_tuple(kDLCPU, gl_devtype)) {
    auto texture = reinterpret_cast<Texture*>(to);
    const void* data = static_cast<const char*>(from) + from_offset;
    auto elemsz = texture->elemsz();
    auto begin = static_cast<GLint>(to_offset / elemsz);
    auto nelems = static_cast<GLsizei>(size / elemsz);
    PutTextureData(texture, begin, nelems, data);

  } else {
    LOG(FATAL) << "Expect copy from/to OpenGL or between OpenGL";
  }
}

void OpenGLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {}

OpenGLWorkspace::OpenGLWorkspace() {
  // Set an error handler.
  // This can be called before glfwInit().
  glfwSetErrorCallback(&GlfwErrorCallback);

  // Initialize GLFW.
  if (glfwInit() != GL_TRUE) {
    LOG(FATAL) << "glfwInit() failed!";
  }

  // Create a window.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
  window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "", nullptr, nullptr);
  if (window_ == nullptr) {
    LOG(FATAL) << "glfwCreateWindow() failed!";
  }

  // Before using any OpenGL API, we must specify a context.
  glfwMakeContextCurrent(window_);

  // Load all OpenGL API function pointers.
  gl = std::unique_ptr<GLFunctionPointers>(new GLFunctionPointers);

  CheckOpenGLError();

  // We always render the same vertices and triangles.
  GLuint vertex_buffer;
  OPENGL_CALL(gl->GenBuffers(1, &vertex_buffer));
  OPENGL_CALL(gl->BindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
  OPENGL_CALL(gl->BufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                             GL_STATIC_DRAW));

  GLuint vertex_array;
  OPENGL_CALL(gl->GenVertexArrays(1, &vertex_array));
  OPENGL_CALL(gl->BindVertexArray(vertex_array));
  OPENGL_CALL(gl->BindBuffer(GL_ARRAY_BUFFER, vertex_buffer));

  // We always use the same vertex shader.
  vertex_shader_ = CreateShader(GL_VERTEX_SHADER, vertex_shader_text_);

  LOG(INFO) << "OpenGL initialized, version = " << gl->GetString(GL_VERSION);
}

OpenGLWorkspace::~OpenGLWorkspace() {
  // Paired with glfwCreateWindow().
  glfwDestroyWindow(window_);

  // Paired with glfwInit().
  glfwTerminate();
}

void OpenGLWorkspace::BindTextureUnit(GLuint unit, GLuint texture) {
  OPENGL_CALL(gl->ActiveTexture(GL_TEXTURE0 + unit));
  OPENGL_CALL(gl->BindTexture(GL_TEXTURE_2D, texture));
}

void OpenGLWorkspace::OnDeleteTexture(GLuint texture) {
  OPENGL_CALL(gl->DeleteTextures(1, &texture));
}

void OpenGLWorkspace::OnDeleteProgram(GLuint program) {
  OPENGL_CALL(gl->DeleteProgram(program));
}

GLuint OpenGLWorkspace::NumTextureUnits() {
  GLint num_units;
  OPENGL_CALL(gl->GetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &num_units));
  return static_cast<GLuint>(num_units);
}

const OpenGLWorkspace::Vertex OpenGLWorkspace::vertices[OpenGLWorkspace::kNumVertices] = {
    {-1.f, -1.f},
    {1.0f, -1.f},
    {1.0f, 1.0f},
    {-1.f, -1.f},
    {-1.f, 1.0f},
    {1.0f, 1.0f},
};

// Don't need to change this.
// The vertex shader only needs to take in the triangle points.
// No need for point transformations.
const char* OpenGLWorkspace::vertex_shader_text_ = "#version 300 es\n"
    "in vec2 point; // input to vertex shader\n"
    "void main() {\n"
    "  gl_Position = vec4(point, 0.0, 1.0);\n"
    "}\n";

Program OpenGLWorkspace::CreateProgram(
    const char* fragment_shader_src) {
  // Create and compile the shaders.
  GLuint fragment_shader = CreateShader(GL_FRAGMENT_SHADER,
                                        fragment_shader_src);

  // Link the shaders and create the program.
  Program program = CreateProgram(fragment_shader);

  OPENGL_CALL(gl->DeleteShader(fragment_shader));

  return program;
}

GLuint OpenGLWorkspace::CreateShader(GLenum shader_kind,
                                     const char* shader_src) {
  // Create the shader.
  GLuint shader = gl->CreateShader(shader_kind);
  gl->ShaderSource(shader, 1, &shader_src, nullptr);
  gl->CompileShader(shader);

  // Check compile errors.
  GLint err;
  gl->GetShaderiv(shader, GL_COMPILE_STATUS, &err);

  GLint info_log_len;
  gl->GetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_len);

  if (err != GL_TRUE) {
    std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
    gl->GetShaderInfoLog(shader, info_log_len, nullptr, err_msg.get());
    LOG(FATAL) << err_msg.get() << "\n" << shader_src;
    assert(false);
  }

  CheckOpenGLError();

  return shader;
}

static TextureFormat GetTextureFormat(TVMType type) {
  CHECK_EQ(type.lanes, 1) << "Not supporting multi-lane types.";

  switch (type.code) {
    case kDLInt: {
      switch (type.bits) {
        case 8:
          return {GL_R8I, GL_RED_INTEGER, GL_BYTE};
        case 16:
          return {GL_R16I, GL_RED_INTEGER, GL_SHORT};
        case 32:
          return {GL_R32I, GL_RED_INTEGER, GL_INT};
        default:
          LOG(FATAL) << "Unsupported type bits " << type.bits;
      }
    }
    case kDLUInt: {
      switch (type.bits) {
        case 8:
          return {GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE};
        case 16:
          return {GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT};
        case 32:
          return {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT};
        default:
          LOG(FATAL) << "Unsupported type bits " << type.bits;
      }
    }
    case kDLFloat: {
      switch (type.bits) {
        case 32:
          return {GL_R32F, GL_RED, GL_FLOAT};
        default:
          LOG(FATAL) << "Unsupported type bits " << type.bits;
      }
    }
    default: {
      LOG(FATAL) << "Unsupported type code" << type.code;
    }
  }
  return {GL_R32F, GL_RED, GL_FLOAT};
}

Texture OpenGLWorkspace::CreateTexture(TVMType type, size_t nbytes) {
  // Create a texture.
  GLuint texture;
  OPENGL_CALL(gl->GenTextures(1, &texture));

  BindTextureUnit(NumTextureUnits() - 1, texture);

  // Use glTexImage2D with nullptr data to specify GPU data storage.
  auto texture_format = GetTextureFormat(type);
  auto nelems = static_cast<GLsizei>(nbytes / (type.bits / 8));
  auto height = (nelems + kTextureRowSize - 1) / kTextureRowSize;
  auto width = (height == 1) ? nelems : kTextureRowSize;
  OPENGL_CALL(gl->TexImage2D(GL_TEXTURE_2D, /*level=*/0,
                             texture_format.internal_format,
                             width, height, /*border=*/0,
                             texture_format.format, texture_format.type,
                             /*data=*/nullptr));

  OPENGL_CALL(
      gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  OPENGL_CALL(
      gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  OPENGL_CALL(
      gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  OPENGL_CALL(
      gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

  return Texture(this, texture, texture_format, width, height);
}

Program OpenGLWorkspace::CreateProgram(GLuint fragment_shader) {
  // Create the program and link the shaders.
  GLuint program = gl->CreateProgram();
  gl->AttachShader(program, vertex_shader_);
  gl->AttachShader(program, fragment_shader);
  gl->LinkProgram(program);

  // Check link errors.
  GLint err;
  gl->GetProgramiv(program, GL_LINK_STATUS, &err);

  GLint info_log_len;
  gl->GetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_len);

  if (err != GL_TRUE) {
    std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
    gl->GetProgramInfoLog(program, info_log_len, nullptr, err_msg.get());
    LOG(FATAL) << err_msg.get();
    assert(false);
  }

  CheckOpenGLError();

  OPENGL_CALL(gl->DetachShader(program, vertex_shader_));
  OPENGL_CALL(gl->DetachShader(program, fragment_shader));

  auto point_attrib = GLuint(gl->GetAttribLocation(program, "point"));
  OPENGL_CALL(gl->EnableVertexAttribArray(point_attrib));

  OPENGL_CALL(gl->VertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE,
                                      sizeof(Vertex), nullptr));

  return Program(this, program);
}

/*!
 * \brief Visit a 1D range of an OpenGL texture-backed TVM array.
 * When getting/setting a sub image of a texture, we can only specify a 2D
 * block (xbeg, ybeg, width, height).
 * Since we are storing all TVM arrays using (kTextureRowSize x nrows) 2D
 * textures (row-major), a range in an array does not necessarily map to a 2D
 * block.
 * This function split a 1D range into 3 2D blocks.
 * \param beg The index of the first element in the 1D range.
 * \param end The index of the last + 1 element in the 1D range.
 * \param on_2d_block Callback for each 2D block. Must have interface
 * void(GLint xbeg, GLint ybeg, GLsizei width, GLsizei height).
 */
template <typename F>
static void Visit1DRange(GLint beg, GLint end, F&& on_2d_block) {
  CHECK_LE(beg, end) << "Invalid range.";

  //           xbeg         kTextureRowSize
  // ybeg  ....************
  //       ****************
  //       ****************
  // ylast *********.......
  //           xlast
  GLint xbeg = beg % kTextureRowSize;
  GLint ybeg = beg / kTextureRowSize;
  GLint xlast = (end - 1) % kTextureRowSize;
  GLint ylast = (end - 1) / kTextureRowSize;

  if (ybeg == ylast) {  // Only one row.
    on_2d_block(xbeg, ybeg, end - beg, 1);
    return;
  }

  // First row.
  on_2d_block(xbeg, ybeg, kTextureRowSize - xbeg, 1);

  // Middle block.
  if (ylast - ybeg > 1) {
    on_2d_block(0, ybeg + 1, kTextureRowSize, ylast - ybeg - 1);
  }

  // Last row.
  on_2d_block(0, ylast, xlast + 1, 1);
}

void OpenGLWorkspace::PutTextureData(Texture *texture,
                                     GLint begin,
                                     GLsizei nelems,
                                     const GLvoid* data) {
  // Bind to temporary unit.
  BindTextureUnit(NumTextureUnits() - 1, texture->texture());

  Visit1DRange(begin, begin + nelems, [&](GLint xbeg, GLint ybeg,
                                          GLsizei width, GLsizei height) {
    auto offset = (ybeg * kTextureRowSize + xbeg - begin) * texture->elemsz();
    const GLvoid* ptr = static_cast<const char*>(data) + offset;

    // Similar to cudaMemcpy.
    OPENGL_CALL(gl->TexSubImage2D(GL_TEXTURE_2D, /*level=*/0,
                                  xbeg, ybeg, width, height,
                                  texture->format_.format,
                                  texture->format_.type, ptr));
  });
}

void OpenGLWorkspace::GetTextureData(const Texture *texture,
                                     GLint begin,
                                     GLsizei nelems,
                                     GLvoid* data) {
  BindTextureUnit(NumTextureUnits() - 1, texture->texture());

  // Create frame buffer.
  GLuint frame_buffer;
  OPENGL_CALL(gl->GenFramebuffers(1, &frame_buffer));
  OPENGL_CALL(gl->BindFramebuffer(GL_FRAMEBUFFER, frame_buffer));

  // Bind texture to framebuffer's attachment 0.
  OPENGL_CALL(gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_2D, texture->texture(), 0));

  // Always check that our framebuffer is okay.
  if (gl->CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(FATAL) << "Framebuffer not complete.";
  }

#ifdef __EMSCRIPTEN__
  // WebGL2's glReadPixels API doesn't allow GL_RED user buffer format.
  // Instead, We must use GL_RGBA. This means the data we retrieve has useless
  // GBA channels. Here we are applying a dirty hack.
  // TODO(zhixunt): We really want to utilize all RGBA channels in textures.
  //
  // WebGL2's glReadPixels API also doesn't allow GL_RED_INTEGER or
  // GL_RGB_INTEGER user buffer format, which means we cannot retrieve integer
  // texture data? (need to confirm)

  CHECK_EQ(texture->format_.internal_format, GL_R32F)
      << "Retrieving integer texture not supported yet.";
  auto elemsz = texture->format_.elemsz();
  auto nchannels = 4;
  auto padded_data_size = nchannels * nelems * elemsz;
  auto padded_data = std::unique_ptr<char[]>(new char[padded_data_size]);
  Visit1DRange(begin, begin + nelems, [&](GLint xbeg, GLint ybeg,
                                          GLsizei width, GLsizei height) {
    auto data_offset = (ybeg * kTextureRowSize + xbeg - begin) * elemsz;
    auto padded_data_offset = data_offset * nchannels;
    OPENGL_CALL(gl->ReadPixels(xbeg, ybeg, width, height,
                               GL_RGBA, GL_FLOAT,
                               padded_data.get() + padded_data_offset));
  });
  for (GLsizei i = 0; i != nelems; ++i) {
    auto dst = reinterpret_cast<char *>(data) + i * elemsz;
    auto src = padded_data.get() + nchannels * i * elemsz;
    std::memcpy(dst, src, elemsz);
  }
#else
  Visit1DRange(begin, begin + nelems, [&](GLint xbeg, GLint ybeg,
                                          GLsizei width, GLsizei height) {
    auto offset = (ybeg * kTextureRowSize + xbeg - begin) * texture->elemsz();
    GLvoid* ptr = static_cast<char*>(data) + offset;

    OPENGL_CALL(gl->ReadPixels(xbeg, ybeg, width, height,
                               texture->format_.format, texture->format_.type,
                               ptr));
  });
#endif

  OPENGL_CALL(gl->DeleteFramebuffers(1, &frame_buffer));
}

void OpenGLWorkspace::SetCurrentProgram(const Program& program) {
  OPENGL_CALL(gl->UseProgram(program.program()));
}

void OpenGLWorkspace::SetUniform(const Program& program,
                                 const std::string& name,
                                 TVMType type,
                                 void* value) {
  GLint location = gl->GetUniformLocation(program.program(), name.c_str());
  switch (type.code) {
    case kDLInt: {
      CHECK_EQ(type.bits, 32) << "Only support 32-bit int for uniform.";
      GLint uniform_value = *reinterpret_cast<GLint*>(value);
      OPENGL_CALL(gl->Uniform1i(location, uniform_value));
      break;
    }
    case kDLUInt: {
      LOG(FATAL) << "Strangely, emcc WebGL does not support glUniform1ui.";
      break;
    }
    case kDLFloat: {
      CHECK_EQ(type.bits, 32) << "Only support 32-bit float for uniform.";
      GLfloat uniform_value = *reinterpret_cast<GLfloat*>(value);
      OPENGL_CALL(gl->Uniform1f(location, uniform_value));
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported type code for uniform.";
      break;
    }
  }
}

void OpenGLWorkspace::SetInputTexture(const Program& program,
                                      const std::string& name,
                                      GLuint unit,
                                      Texture* texture) {
  // We always use the last texture unit as temporary.
  // Therefore, we can have "NumTextureUnits() - 1" input textures.
  CHECK_LT(unit, NumTextureUnits() - 1) << "Too many textures.";

  BindTextureUnit(unit, texture->texture());
  GLint location = gl->GetUniformLocation(program.program_, name.c_str());
  OPENGL_CALL(gl->Uniform1i(location, unit));
}

void OpenGLWorkspace::Render(Texture* output) {
  // Create frame buffer.
  GLuint frame_buffer;
  OPENGL_CALL(gl->GenFramebuffers(1, &frame_buffer));
  OPENGL_CALL(gl->BindFramebuffer(GL_FRAMEBUFFER, frame_buffer));

  // Set "renderedTexture" as our colour attachement 0.
  OPENGL_CALL(gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_2D, output->texture(), 0));

  // Specify that we will render to color attachment 0.
  GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
  OPENGL_CALL(gl->DrawBuffers(1, DrawBuffers));

  // Always check that our framebuffer is okay.
  if (gl->CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(FATAL) << "Framebuffer not complete.";
  }

  // Perform rendering.
  OPENGL_CALL(gl->Viewport(0, 0, output->width(), output->height()));
  OPENGL_CALL(gl->Clear(GL_COLOR_BUFFER_BIT));
  OPENGL_CALL(gl->DrawArrays(GL_TRIANGLES, 0, 6));

  OPENGL_CALL(gl->DeleteFramebuffers(1, &frame_buffer));
}

TVM_REGISTER_GLOBAL("device_api.opengl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = OpenGLWorkspace::Global().get();
  *rv = static_cast<void*>(ptr);
});

}  // namespace gl
}  // namespace runtime
}  // namespace tvm
