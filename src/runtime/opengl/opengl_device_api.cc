/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_device_api.cc
 */
#include "./opengl_common.h"

#if TVM_OPENGL_RUNTIME

#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace gl {

inline const char* GLGetErrorString(GLenum error) {
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

void OPENGL_CHECK_ERROR() {
  GLenum err = glGetError();
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
    OPENGL_CHECK_ERROR();                                                      \
  }

void GlfwErrorCallback(int err, const char* str) {
  LOG(ERROR) << "Error: [" << err << "] " << str;
}

// Always only use the first dimension of a 2D texture.
// The reason of using 2D textures is that texelFetch only supports 2D textures.
Texture::Texture(size_t nbytes)
    : texture_(kInvalidTexture),
      width_(static_cast<decltype(width_)>(nbytes / sizeof(GLfloat))),
      height_(1) {
  LOG(INFO) << "Created texture [" << texture_ << "]";
  CHECK((nbytes % sizeof(GLfloat)) == 0) << "Must be multiple of GLfloats";

  // Create a texture.
  OPENGL_CALL(glGenTextures(1, &texture_));

  auto workspace = gl::OpenGLWorkspace::Global();
  workspace->BindTextureUnit(workspace->NumTextureUnits() - 1, texture_);

  // Use glTexImage2D with nullptr data to specify GPU data storage.
  // TODO(pengw): How can we know the type of data?
  OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_R32F,
                           width_, /*height=*/1, /*border=*/0,
                           GL_RED, GL_FLOAT, /*data=*/nullptr));

  // TODO(zhixunt): What are these?
  OPENGL_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  OPENGL_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  OPENGL_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  OPENGL_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
}

Texture::~Texture() {
  if (texture_ != kInvalidTexture) {
    LOG(INFO) << "Deleting texture [" << texture_ << "]";

    OPENGL_CALL(glDeleteTextures(1, &texture_));
    texture_ = kInvalidTexture;
  }
}

void Texture::GetData(GLvoid* data) const {
  // Bind to temporary unit.
  auto workspace = gl::OpenGLWorkspace::Global();
  workspace->BindTextureUnit(workspace->NumTextureUnits() - 1, this->texture_);

  glGetTexImage(GL_TEXTURE_2D, /*level=*/0, GL_RED, GL_FLOAT, data);
}

void Texture::PutData(GLint begin, GLsizei nelems, const GLvoid* data) {
  LOG(INFO) << "Texture::PutData(" << "begin = " << begin << ", "
                    << "nelems = " << nelems << ", data)";

  // Bind to temporary unit.
  auto workspace = gl::OpenGLWorkspace::Global();
  workspace->BindTextureUnit(workspace->NumTextureUnits() - 1, this->texture_);

  // Similar to cudaMemcpy.
  OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, /*level=*/0,
                              /*xoffset=*/begin, /*yoffset=*/0,
                              /*width=*/nelems, /*height=*/1,
                              GL_RED, GL_FLOAT, data));
}

Program::~Program() {
  if (program_ != kInvalidProgram) {
    glDeleteProgram(program_);
    program_ = kInvalidProgram;
  }
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
    TVMContext ctx, size_t nbytes, size_t alignment) {
  LOG(INFO)
      << "OpenGLWorkspace::AllocDataSpace(ctx, nbytes = "
      << nbytes << ", alignment = " << alignment << ")";
  return reinterpret_cast<void*>(new Texture(nbytes));
}

void OpenGLWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  LOG(INFO)
      << "OpenGLWorkspace::FreeDataSpace(ctx, ptr = "
      << ptr << ")";
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

  if (ctx_from.device_type == gl_devtype && ctx_to.device_type == gl_devtype) {
    // OpenGL texture => OpenGL texture

    // TODO(pengw): Implement this.
    LOG(FATAL) << "Not Implemented";

  } else if (ctx_from.device_type == gl_devtype &&
      ctx_to.device_type == kDLCPU) {
    // OpenGL texture => CPU memory buffer

    auto texture = static_cast<const Texture*>(from);
    CHECK(from_offset == 0U &&
          size == static_cast<size_t>(texture->width()) * sizeof(GLfloat))
      << "Only support full texture retrieval.";

    texture->GetData(static_cast<char *>(to) + to_offset);

  } else if (ctx_from.device_type == kDLCPU &&
             ctx_to.device_type == gl_devtype) {
    // CPU memory buffer => OpenGL texture

    CHECK(to_offset % sizeof(GLfloat) == 0) << "Must be multiple of GLfloats.";
    CHECK(size % sizeof(GLfloat) == 0) << "Must be multiple of GLfloats.";

    auto texture = reinterpret_cast<Texture*>(to);
    const void* data = static_cast<const char*>(from) + from_offset;
    auto begin = static_cast<GLint>(to_offset / sizeof(GLfloat));
    auto nelems = static_cast<GLsizei>(size / sizeof(GLfloat));
    texture->PutData(begin, nelems, data);

  } else {
    LOG(FATAL) << "Expect copy from/to OpenGL or between OpenGL";
  }
}

void OpenGLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  // TODO(zhixunt): Implement this.
  LOG(INFO) << "OpenGLWorkspace::StreamSync";
}

void* OpenGLWorkspace::AllocWorkspace(TVMContext ctx, size_t size) {
  // TODO(zhixunt): Implement this.
  LOG(INFO) << "OpenGLWorkspace::AllocWorkspace";
  return nullptr;
}

void OpenGLWorkspace::FreeWorkspace(TVMContext ctx, void* data) {
  // TODO(zhixunt): Implement this.
  LOG(INFO) << "OpenGLWorkspace::FreeWorkspace";
}

OpenGLWorkspace::OpenGLWorkspace() {
  // Set an error handler.
  // This can be called before glfwInit().
  glfwSetErrorCallback(&GlfwErrorCallback);

  // Initialize GLFW.
  if (glfwInit() != GL_TRUE) {
    LOG(ERROR) << "glfwInit() failed!";
    assert(false);
  }

  // Create a window.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
  window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "", nullptr, nullptr);
  if (window_ == nullptr) {
    LOG(ERROR) << "glfwCreateWindow() failed!";
    assert(false);
  }

  LOG(INFO) << "GLFW says OpenGL version: "
                    << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MAJOR)
                    << "."
                    << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MINOR)
                    << "."
                    << glfwGetWindowAttrib(window_, GLFW_CONTEXT_REVISION);

  // Before using any OpenGL API, we must specify a context.
  glfwMakeContextCurrent(window_);

  // Must be called after creating GLFW window.
  gladLoadGL();

  LOG(INFO) << "OpenGL says version: " << glGetString(GL_VERSION);

  OPENGL_CHECK_ERROR();

  // We always render the same vertices and triangles.
  GLuint vertex_buffer;
  OPENGL_CALL(glGenBuffers(1, &vertex_buffer));
  OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
  OPENGL_CALL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                           GL_STATIC_DRAW));

  GLuint vertex_array;
  OPENGL_CALL(glGenVertexArrays(1, &vertex_array));
  OPENGL_CALL(glBindVertexArray(vertex_array));
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

  // We always use the same vertex shader.
  vertex_shader_ = CreateShader(GL_VERTEX_SHADER, vertex_shader_text_);
}

OpenGLWorkspace::~OpenGLWorkspace() {
  LOG(INFO) << "~OpenGLWorkspace()";
  // Paired with glfwCreateWindow().
  glfwDestroyWindow(window_);
  // Paired with glfwInit().
  glfwTerminate();
}

void OpenGLWorkspace::BindTextureUnit(GLuint unit, GLuint texture) {
  OPENGL_CALL(glActiveTexture(GL_TEXTURE0 + unit));
  OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
}

GLuint OpenGLWorkspace::NumTextureUnits() {
  GLint num_units;
  OPENGL_CALL(glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &num_units));
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
const char* OpenGLWorkspace::vertex_shader_text_ = "#version 330 core\n"
    "in vec2 point; // input to vertex shader\n"
    "void main() {\n"
    "  gl_Position = vec4(point, 0.0, 1.0);\n"
    "}\n";

std::unique_ptr<Program> OpenGLWorkspace::CreateProgram(
    const char* fragment_shader_src) {
  // Create and compile the shaders.
  GLuint fragment_shader = CreateShader(GL_FRAGMENT_SHADER,
                                        fragment_shader_src);

  // Link the shaders and create the program.
  auto program = CreateProgram(fragment_shader);

  OPENGL_CALL(glDeleteShader(fragment_shader));

  return program;
}

GLuint OpenGLWorkspace::CreateShader(GLenum shader_kind,
                                     const char* shader_src) {
  // Create the shader.
  GLuint shader = glCreateShader(shader_kind);
  glShaderSource(shader, 1, &shader_src, nullptr);
  glCompileShader(shader);

  // Check compile errors.
  GLint err;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &err);

  GLint info_log_len;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_len);

  if (info_log_len > 0) {
    std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
    glGetShaderInfoLog(shader, info_log_len, nullptr, err_msg.get());
    LOG(ERROR) << err_msg.get();
    assert(false);
  }

  OPENGL_CHECK_ERROR();

  return shader;
}

std::unique_ptr<Program> OpenGLWorkspace::CreateProgram(
    GLuint fragment_shader) {
  // Create the program and link the shaders.
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex_shader_);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);

  // Check link errors.
  GLint err;
  glGetProgramiv(program, GL_LINK_STATUS, &err);

  GLint info_log_len;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_len);

  if (info_log_len > 0) {
    std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
    glGetProgramInfoLog(program, info_log_len, nullptr, err_msg.get());
    LOG(ERROR) << err_msg.get();
    assert(false);
  }

  OPENGL_CHECK_ERROR();

  OPENGL_CALL(glDetachShader(program, vertex_shader_));
  OPENGL_CALL(glDetachShader(program, fragment_shader));

  auto point_attrib = GLuint(glGetAttribLocation(program, "point"));
  OPENGL_CALL(glEnableVertexAttribArray(point_attrib));

  OPENGL_CALL(glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE,
                                    sizeof(Vertex), nullptr));

  return std::unique_ptr<Program>(new Program(program));
}

void OpenGLWorkspace::Render(
    const Program& program,
    const std::vector<std::pair<std::string, Texture*>>& inputs,
    Texture* output) {
  if (inputs.size() + 2 > NumTextureUnits()) {
    LOG(ERROR) << "Too many inputs!";
    assert(false);
  }

  OPENGL_CALL(glUseProgram(program.program_));

  // Create frame buffer.
  GLuint frame_buffer;
  OPENGL_CALL(glGenFramebuffers(1, &frame_buffer));
  OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));

  // Set "renderedTexture" as our colour attachement #0
  OPENGL_CALL(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   output->texture(), 0));

  // Set the list of draw buffers.
  GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
  // "1" is the size of DrawBuffers.
  OPENGL_CALL(glDrawBuffers(1, DrawBuffers));

  // Always check that our framebuffer is ok
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(ERROR) << "Framebuffer not complete.";
    assert(false);
  }

  // Tell the fragment shader what input textures to use.
  for (GLuint unit = 0; unit != inputs.size(); ++unit) {
    const std::string& name = inputs[unit].first;
    auto texture = inputs[unit].second;

    BindTextureUnit(unit, texture->texture());

    GLint texture_uniform = glGetUniformLocation(program.program_,
                                                 name.c_str());
    OPENGL_CALL(glUniform1i(texture_uniform, unit));
  }

  OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
  OPENGL_CALL(glViewport(0, 0, output->width(), output->height()));

  OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
  OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));

  glDeleteFramebuffers(1, &frame_buffer);
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
