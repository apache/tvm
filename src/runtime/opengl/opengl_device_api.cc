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

void GlfwErrorCallback(int err, const char* str) {
  LOG(ERROR) << "Error: [" << err << "] " << str;
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
  auto texture = CreateTexture(nbytes).release();
  return reinterpret_cast<void*>(texture);
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

    GetTextureData(texture, static_cast<char *>(to) + to_offset);

  } else if (ctx_from.device_type == kDLCPU &&
             ctx_to.device_type == gl_devtype) {
    // CPU memory buffer => OpenGL texture

    CHECK(to_offset % sizeof(GLfloat) == 0) << "Must be multiple of GLfloats.";
    CHECK(size % sizeof(GLfloat) == 0) << "Must be multiple of GLfloats.";

    auto texture = reinterpret_cast<Texture*>(to);
    const void* data = static_cast<const char*>(from) + from_offset;
    auto begin = static_cast<GLint>(to_offset / sizeof(GLfloat));
    auto nelems = static_cast<GLsizei>(size / sizeof(GLfloat));
    PutTextureData(texture, begin, nelems, data);

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

  LOG(INFO) << "Calling gladLoadGL...";

  gl = std::unique_ptr<GLFunctionPointers>(new GLFunctionPointers);

  // Must be called after creating GLFW window.
//  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

  LOG(INFO) << "OpenGL says version: " << glGetString(GL_VERSION);

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

std::unique_ptr<Program> OpenGLWorkspace::CreateProgram(
    const char* fragment_shader_src) {
  // Create and compile the shaders.
  GLuint fragment_shader = CreateShader(GL_FRAGMENT_SHADER,
                                        fragment_shader_src);

  // Link the shaders and create the program.
  auto program = CreateProgram(fragment_shader);

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

std::unique_ptr<Texture> OpenGLWorkspace::CreateTexture(size_t nbytes) {
  CHECK((nbytes % sizeof(GLfloat)) == 0) << "Must be multiple of GLfloats";

  // Create a texture.
  GLuint texture;
  OPENGL_CALL(gl->GenTextures(1, &texture));

  BindTextureUnit(NumTextureUnits() - 1, texture);

  // Use glTexImage2D with nullptr data to specify GPU data storage.
  // TODO(pengw): How can we know the type of data?
  auto width = static_cast<GLsizei>(nbytes / sizeof(GLfloat));
  auto height = GLsizei(1);
  OPENGL_CALL(gl->TexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_R32F,
                             width, height, /*border=*/0,
                             GL_RED, GL_FLOAT, /*data=*/nullptr));

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

  return std::unique_ptr<Texture>(new Texture(this, texture, width, height));
}

std::unique_ptr<Program> OpenGLWorkspace::CreateProgram(
    GLuint fragment_shader) {
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

  return std::unique_ptr<Program>(new Program(this, program));
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
                                GL_RED, GL_FLOAT, data));
}

void OpenGLWorkspace::GetTextureData(const Texture *texture, GLvoid* data) {
  BindTextureUnit(NumTextureUnits() - 1, texture->texture());

  // Create frame buffer.
  GLuint frame_buffer;
  OPENGL_CALL(gl->GenFramebuffers(1, &frame_buffer));
  OPENGL_CALL(gl->BindFramebuffer(GL_FRAMEBUFFER, frame_buffer));

  // Bind texture to framebuffer's attachment #0.
  OPENGL_CALL(gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_2D, texture->texture(), 0));

  // Always check that our framebuffer is ok
  if (gl->CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(ERROR) << "Framebuffer not complete.";
    assert(false);
  }

  OPENGL_CALL(gl->ReadPixels(/*x=*/0, /*y=*/0, /*width=*/texture->width(),
                             /*height=*/texture->height(), GL_RED, GL_FLOAT,
                             data));

  OPENGL_CALL(gl->DeleteFramebuffers(1, &frame_buffer));
}

void OpenGLWorkspace::Render(
    const Program& program,
    const std::vector<std::pair<std::string, Texture*>>& inputs,
    Texture* output) {
  if (inputs.size() + 2 > NumTextureUnits()) {
    LOG(ERROR) << "Too many inputs!";
    assert(false);
  }

  OPENGL_CALL(gl->UseProgram(program.program_));

  // Create frame buffer.
  GLuint frame_buffer;
  OPENGL_CALL(gl->GenFramebuffers(1, &frame_buffer));
  OPENGL_CALL(gl->BindFramebuffer(GL_FRAMEBUFFER, frame_buffer));

  // Set "renderedTexture" as our colour attachement #0
  OPENGL_CALL(gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_2D, output->texture(), 0));

  // Set the list of draw buffers.
  GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
  // "1" is the size of DrawBuffers.
  OPENGL_CALL(gl->DrawBuffers(1, DrawBuffers));

  // Always check that our framebuffer is ok
  if (gl->CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(ERROR) << "Framebuffer not complete.";
    assert(false);
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

  OPENGL_CALL(gl->BindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
  OPENGL_CALL(gl->Viewport(0, 0, output->width(), output->height()));

  OPENGL_CALL(gl->Clear(GL_COLOR_BUFFER_BIT));
  OPENGL_CALL(gl->DrawArrays(GL_TRIANGLES, 0, 6));

  gl->DeleteFramebuffers(1, &frame_buffer);
}

TVM_REGISTER_GLOBAL("device_api.opengl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = OpenGLWorkspace::Global().get();
  *rv = static_cast<void*>(ptr);
});

}  // namespace gl

void TestLocalOpenGL() {
  std::cout << "Calling TestLocalOpenGL()" << std::endl;

  std::shared_ptr<gl::OpenGLWorkspace> workspace = gl::OpenGLWorkspace::Global();

  std::cout << "Got workspace" << std::endl;

  const char* shader_src = "#version 300 es\n"
    "precision highp float;\n"
    "out float color;\n"
    "void main() {\n"
    "  color = 0.0;\n"
    "}\n";

  std::unique_ptr<gl::Program> program = workspace->CreateProgram(shader_src);

  std::cout << "Created program" << std::endl;

  std::unique_ptr<gl::Texture> output = workspace->CreateTexture(16);

  std::cout << "Created texture" << std::endl;

  workspace->Render(*program, {}, output.get());

  std::cout << "Rendered" << std::endl;
}

TVM_REGISTER_GLOBAL("contrib.rpc._TestLocalOpenGL")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  TestLocalOpenGL();
  *rv = nullptr;
});

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENGL_RUNTIME
