/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_device_api.cc
 */
#include "./opengl_common.h"

#if TVM_OPENGL_RUNTIME

#include <tvm/runtime/registry.h>
#include <cstring>

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
    case GL_STACK_OVERFLOW:
      return "GL_STACK_OVERFLOW";
    case GL_STACK_UNDERFLOW:
      return "GL_STACK_UNDERFLOW";
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
  CHECK(err == GL_NO_ERROR) << "OpenGL error, code=" << err << ": "
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
  // TODO(zhixunt): Implement this.
  LOG(INFO) << "OpenGLWorkspace::SetDevice";
}

void OpenGLWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  // TODO(zhixunt): Implement this.
  LOG(INFO) << "OpenGLWorkspace::GetAttr";
}

void* OpenGLWorkspace::AllocDataSpace(
    TVMContext ctx, size_t nbytes, size_t alignment, TVMType type_hint) {
  LOG(INFO) << "OpenGLWorkspace::AllocDataSpace(ctx, nbytes = " << nbytes
            << ", alignment = " << alignment << ")";
  return reinterpret_cast<void*>(new Texture(CreateTexture(type_hint, nbytes)));
}

void OpenGLWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  LOG(INFO) << "OpenGLWorkspace::FreeDataSpace(ctx, ptr = " << ptr << ")";
  delete reinterpret_cast<Texture*>(ptr);
}

void OpenGLWorkspace::CopyDataFromTo(const void* from,
                                     size_t from_offset,
                                     void* to,
                                     size_t to_offset,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMStreamHandle stream) {
  LOG(INFO)
      << "OpenGLWorkspace::CopyDataFromTo("
      << "from = " << from << ", "
      << "from_offset = " << from_offset << ", "
      << "to = " << to << ", "
      << "to_offset = " << to_offset << ", "
      << "size = " << size << ", "
      << "ctx_from = (" << ctx_from.device_type << ", " << ctx_from.device_id
      << "), "
      << "ctx_to = (" << ctx_to.device_type << ", " << ctx_to.device_id
      << "), stream)";
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

void OpenGLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  // TODO(zhixunt): Implement this.
  LOG(INFO) << "OpenGLWorkspace::StreamSync()";
}

void* OpenGLWorkspace::AllocWorkspace(TVMContext ctx, size_t size) {
  LOG(FATAL) << "Cannot allocate OpenGL workspace.";
  return nullptr;
}

void OpenGLWorkspace::FreeWorkspace(TVMContext ctx, void* data) {
  LOG(FATAL) << "Cannot free OpenGL workspace.";
}

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

  LOG(INFO) << "GLFW says OpenGL version: "
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MAJOR)
            << "."
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MINOR)
            << "."
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_REVISION);

  // Before using any OpenGL API, we must specify a context.
  glfwMakeContextCurrent(window_);

  // Load all OpenGL API function pointers.
  gl = std::unique_ptr<GLFunctionPointers>(new GLFunctionPointers);

  LOG(INFO) << "OpenGL says version: " << gl->GetString(GL_VERSION);

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
  LOG(INFO) << "Created vertex shader";
}

OpenGLWorkspace::~OpenGLWorkspace() {
  LOG(INFO) << "~OpenGLWorkspace()";

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
    LOG(ERROR) << err_msg.get();
    assert(false);
  }

  CheckOpenGLError();

  return shader;
}

static TextureFormat GetTextureFormat(TVMType type) {
  // TODO(zhixunt) Might want to support this.
  CHECK(type.lanes == 1) << "Currently not supporting multi-lane types.";

  switch (type.code) {
    case kDLInt: {
      switch (type.bits) {
        case 8:
          LOG(INFO) << "Texture data type: int8";
          return {GL_R8I, GL_RED_INTEGER, GL_BYTE};
        case 16:
          LOG(INFO) << "Texture data type: int16";
          return {GL_R16I, GL_RED_INTEGER, GL_SHORT};
        case 32:
          LOG(INFO) << "Texture data type: int32";
          return {GL_R32I, GL_RED_INTEGER, GL_INT};
        default:
          LOG(FATAL) << "Unsupported type bits " << type.bits;
      }
    }
    case kDLUInt: {
      switch (type.bits) {
        case 8:
          LOG(INFO) << "Texture data type: uint8";
          return {GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE};
        case 16:
          LOG(INFO) << "Texture data type: uint16";
          return {GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT};
        case 32:
          LOG(INFO) << "Texture data type: uint32";
          return {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT};
        default:
          LOG(FATAL) << "Unsupported type bits " << type.bits;
      }
    }
    case kDLFloat: {
      switch (type.bits) {
        case 32:
          LOG(INFO) << "Texture data type: float32";
          return {GL_R32F, GL_RED, GL_FLOAT};
        default:
          LOG(FATAL) << "Unsupported type bits " << type.bits;
      }
    }
    default:
      LOG(FATAL) << "Unsupported type code" << type.code;
  }
  assert(false);
}

Texture OpenGLWorkspace::CreateTexture(TVMType type, size_t nbytes) {
  // Create a texture.
  GLuint texture;
  OPENGL_CALL(gl->GenTextures(1, &texture));

  BindTextureUnit(NumTextureUnits() - 1, texture);

  // Use glTexImage2D with nullptr data to specify GPU data storage.
  auto texture_format = GetTextureFormat(type);
  auto width = static_cast<GLsizei>(nbytes / (type.bits / 8));
  auto height = GLsizei(1);
  OPENGL_CALL(gl->TexImage2D(GL_TEXTURE_2D, /*level=*/0,
                             texture_format.internal_format,
                             width, height, /*border=*/0,
                             texture_format.format, texture_format.type,
                             /*data=*/nullptr));

  // TODO(zhixunt): What are these?
  OPENGL_CALL(
      gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  OPENGL_CALL(
      gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  OPENGL_CALL(
      gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  OPENGL_CALL(
      gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

  LOG(INFO) << "Created texture [" << texture << "]";

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
    LOG(ERROR) << err_msg.get();
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

void OpenGLWorkspace::PutTextureData(Texture *texture, GLint begin,
                                     GLsizei nelems, const GLvoid* data) {
  LOG(INFO) << "Texture::PutData(" << "begin = " << begin << ", "
            << "nelems = " << nelems << ", data)";

  // Bind to temporary unit.
  BindTextureUnit(NumTextureUnits() - 1, texture->texture());

  // Similar to cudaMemcpy.
  OPENGL_CALL(gl->TexSubImage2D(GL_TEXTURE_2D, /*level=*/0,
                                /*xoffset=*/begin, /*yoffset=*/0,
                                /*width=*/nelems, /*height=*/1,
                                texture->format_.format, texture->format_.type,
                                data));
}

void OpenGLWorkspace::GetTextureData(const Texture *texture, GLint begin,
                                     GLsizei nelems, GLvoid* data) {
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

  CHECK(texture->format_.internal_format == GL_R32F)
    << "Retrieving integer texture not supported yet.";
  auto elemsz = texture->format_.elemsz();
  auto nchannels = 4;
  auto padded_data_size = nchannels * nelems * elemsz;
  auto padded_data = std::unique_ptr<char[]>(new char[padded_data_size]);
  OPENGL_CALL(gl->ReadPixels(/*x=*/begin, /*y=*/0, /*width=*/nelems,
                             /*height=*/1, GL_RGBA, GL_FLOAT,
                             padded_data.get()));
  for (GLsizei i = 0; i != nelems; ++i) {
    auto dst = reinterpret_cast<char *>(data) + i * elemsz;
    auto src = padded_data.get() + nchannels * i * elemsz;
    std::memcpy(dst, src, elemsz);
  }
#else
  OPENGL_CALL(gl->ReadPixels(/*x=*/begin, /*y=*/0, /*width=*/nelems,
                             /*height=*/1, texture->format_.format,
                             texture->format_.type, data));
#endif

  OPENGL_CALL(gl->DeleteFramebuffers(1, &frame_buffer));
}

void OpenGLWorkspace::Render(
    const Program& program,
    const std::vector<std::tuple<std::string, TVMType, void*>>& uniforms,
    const std::vector<std::pair<std::string, Texture*>>& inputs,
    Texture* output) {
  if (inputs.size() + 2 > NumTextureUnits()) {
    LOG(FATAL) << "Too many inputs!";
  }

  OPENGL_CALL(gl->UseProgram(program.program_));

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

  // Set up uniforms.
  for (auto &uniform : uniforms) {
    std::string name;
    TVMType type;
    void *value;
    std::tie(name, type, value) = uniform;
    CHECK_EQ(type.lanes, 1) << "Only support scalar uniform.";

    GLint location = gl->GetUniformLocation(program.program_, name.c_str());

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

  // Tell the fragment shader what input textures to use.
  for (GLuint unit = 0; unit != inputs.size(); ++unit) {
    const std::string& name = inputs[unit].first;
    auto texture = inputs[unit].second;

    BindTextureUnit(unit, texture->texture());

    GLint texture_uniform = gl->GetUniformLocation(program.program_,
                                                   name.c_str());
    OPENGL_CALL(gl->Uniform1i(texture_uniform, unit));
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

#endif  // TVM_OPENGL_RUNTIME
