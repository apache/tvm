/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_device_api.cc
 */
#include "./opengl_common.h"

#if TVM_OPENGL_RUNTIME

#include <tvm/runtime/registry.h>
#include <dlpack/dlpack.h>

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
  LOG_ERROR.stream() << "Error: [" << err << "] " << str;
}

// TODO(pengw): How to determine 2D size?
Texture::Texture(size_t size)
    : texture_(kInvalidTexture), width_((GLuint) (size / sizeof(GLfloat))),
      height_(1) {
  // Create a texture.
  OPENGL_CALL(glGenTextures(1, &texture_));
  LOG_INFO.stream() << "Created texture [" << texture_ << "]";
}

Texture::~Texture() {
  if (texture_ != kInvalidTexture) {
    LOG_INFO.stream() << "Deleting texture [" << texture_ << "]";
    OPENGL_CALL(glDeleteTextures(1, &texture_));
    texture_ = kInvalidTexture;
  }
}

void Texture::GetData(GLfloat* data) const {
  auto workspace = gl::OpenGLWorkspace::Global();
  workspace->BindTextureUnit(workspace->NumTextureUnits() - 1, texture_);
  glGetTexImage(GL_TEXTURE_2D, /*level=*/0, GL_RED, GL_FLOAT, data);
}

void Texture::PutData(size_t size, const void* data) {
  auto workspace = gl::OpenGLWorkspace::Global();
  // Bind to temporary unit.
  workspace->BindTextureUnit(workspace->NumTextureUnits() - 1, this->texture_);
  // Similar to cudaMemcpy.
  // TODO(pengw): How can we know the type of data?
  OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_R32F,
                           (GLsizei) size / sizeof(GLfloat), 1, /*border=*/0,
                           GL_RED, GL_FLOAT, data));
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

Program::~Program() {
  if (program_ != kInvalidProgram) {
    glDeleteProgram(program_);
    program_ = kInvalidProgram;
  }
}

const std::shared_ptr<OpenGLWorkspace>& OpenGLWorkspace::Global() {
  static std::shared_ptr<OpenGLWorkspace> inst = std::make_shared<OpenGLWorkspace>();
  return inst;
}

void OpenGLWorkspace::SetDevice(TVMContext ctx) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::SetDevice";
}

void OpenGLWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::GetAttr";
}

void* OpenGLWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment) {
  LOG_INFO.stream()
      << "OpenGLWorkspace::AllocDataSpace(ctx, size = "
      << size << ", alignment = " << alignment << ")";
  this->Init();
  return new Texture(size);
}

void OpenGLWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  LOG_INFO.stream()
      << "OpenGLWorkspace::FreeDataSpace(ctx, ptr = "
      << ptr << ")";
  delete (static_cast<Texture*>(ptr));
}

void OpenGLWorkspace::CopyDataFromTo(const void* from,
                                     size_t from_offset,
                                     void* to,
                                     size_t to_offset,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMStreamHandle stream) {
  LOG_INFO.stream()
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
  this->Init();
  CHECK(stream == nullptr);
  if (ctx_from.device_type == kDLOpenGL && ctx_to.device_type == kDLOpenGL) {
    // TODO(pengw): Implement this.
  } else if (ctx_from.device_type == kDLOpenGL &&
             ctx_to.device_type == kDLCPU) {
    auto texture = static_cast<const Texture*>(from);
    texture->GetData(static_cast<GLfloat*>(to));
  } else if (ctx_from.device_type == kDLCPU &&
             ctx_to.device_type == kDLOpenGL) {
    auto texture = static_cast<Texture*>(to);
    texture->PutData(size, from);
  } else {
    LOG(FATAL) << "Expect copy from/to OpenGL or between OpenGL";
  }
}

void OpenGLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::StreamSync";
}

void* OpenGLWorkspace::AllocWorkspace(TVMContext ctx, size_t size) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::AllocWorkspace";
  return nullptr;
}

void OpenGLWorkspace::FreeWorkspace(TVMContext ctx, void* data) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::FreeWorkspace";
}

void OpenGLWorkspace::Init() {
  if (initialized_) return;
  std::lock_guard<std::mutex>(this->mu);
  if (initialized_) return;
  initialized_ = true;
// Set an error handler.
  // This can be called before glfwInit().
  glfwSetErrorCallback(&GlfwErrorCallback);

  // Initialize GLFW.
  if (glfwInit() != GL_TRUE) {
    LOG_ERROR.stream() << "glfwInit() failed!";
    assert(false);
  }

  // Create a window.
  // TODO(zhixunt): GLFW allows us to create an invisible window.
  // TODO(zhixunt): On retina display, window size is different from framebuffer size.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "", nullptr, nullptr);
  if (window_ == nullptr) {
    LOG_ERROR.stream() << "glfwCreateWindow() failed!";
    assert(false);
  }

  LOG_INFO.stream() << "GLFW says OpenGL version: "
                    << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MAJOR)
                    << "."
                    << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MINOR)
                    << "."
                    << glfwGetWindowAttrib(window_, GLFW_CONTEXT_REVISION);

  // Before using any OpenGL API, we must specify a context.
  glfwMakeContextCurrent(window_);

  // Must be called after creating GLFW window.
  gladLoadGL();

  LOG_INFO.stream() << "OpenGL says version: " << glGetString(GL_VERSION);

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
  LOG_INFO.stream() << "~OpenGLWorkspace()";
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

/*!
 * \brief Create a program that uses the given vertex and fragment shader.
 * \param fragment_shader The fragment shader **source**.
 * \return The program ID.
 */
std::shared_ptr<Program> OpenGLWorkspace::CreateProgram(
    const char* fragment_shader_src) {
  this->Init();
  // Create and compile the shaders.
  GLuint fragment_shader = CreateShader(GL_FRAGMENT_SHADER,
                                        fragment_shader_src);

  // Link the shaders and create the program.
  auto program = CreateProgram(fragment_shader);

  OPENGL_CALL(glDeleteShader(fragment_shader));

  return program;
}

/*!
 * \brief Create and compile a shader from a source string.
 * \param shader_kind The kind of shader.
 * Could be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
 * \param shader_src The source string of the shader.
 * \return The compiled shader ID.
 */
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
    LOG_ERROR.stream() << err_msg.get();
    assert(false);
  }

  OPENGL_CHECK_ERROR();

  return shader;
}

/*!
 * \brief Create a program that uses the given vertex and fragment shaders.
 * \param fragment_shader The **compiled** fragment shader.
 * \return The program ID.
 */
std::shared_ptr<Program> OpenGLWorkspace::CreateProgram(
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
    LOG_ERROR.stream() << err_msg.get();
    assert(false);
  }

  OPENGL_CHECK_ERROR();

  OPENGL_CALL(glDetachShader(program, vertex_shader_));
  OPENGL_CALL(glDetachShader(program, fragment_shader));

  auto point_attrib = GLuint(glGetAttribLocation(program, "point"));
  OPENGL_CALL(glEnableVertexAttribArray(point_attrib));

  OPENGL_CALL(glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE,
                                    sizeof(Vertex), nullptr));

  return std::make_shared<Program>(program);
}

void OpenGLWorkspace::Render(
    const Program& program,
    const std::vector<std::pair<std::string, Texture*>>& inputs,
    Texture* output) {
  if (inputs.size() + 2 > NumTextureUnits()) {
    LOG_ERROR.stream() << "Too many inputs!";
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
    LOG_ERROR.stream() << "Framebuffer not complete.";
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
