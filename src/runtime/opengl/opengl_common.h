/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_common.h
 * \brief OpenGL common header
 */
#ifndef TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_
#define TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <dmlc/logging.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace gl {

// This file contains the following classes.
class GLFunctionPointers;
class OpenGLWorkspace;
class Texture;
class Program;

inline GLFWglproc GetProcAddress(const char* procname) {
  GLFWglproc proc = glfwGetProcAddress(procname);
  CHECK(proc != nullptr) << "Cannot get function \"" << procname << "\"";
  return proc;
}

#define SetGLFunctionPointer(NAME) \
  NAME(decltype(NAME)(GetProcAddress("gl" #NAME)))

/*!
 * \brief The function pointers of all OpenGL APIs that are used.
 * Must be constructed after creating an OpenGL context.
 */
class GLFunctionPointers {
 public:
  GLFunctionPointers()
      : SetGLFunctionPointer(ActiveTexture),
        SetGLFunctionPointer(AttachShader),
        SetGLFunctionPointer(BindBuffer),
        SetGLFunctionPointer(BindFramebuffer),
        SetGLFunctionPointer(BindTexture),
        SetGLFunctionPointer(BindVertexArray),
        SetGLFunctionPointer(BufferData),
        SetGLFunctionPointer(CheckFramebufferStatus),
        SetGLFunctionPointer(Clear),
        SetGLFunctionPointer(CompileShader),
        SetGLFunctionPointer(CreateProgram),
        SetGLFunctionPointer(CreateShader),
        SetGLFunctionPointer(DeleteFramebuffers),
        SetGLFunctionPointer(DeleteProgram),
        SetGLFunctionPointer(DeleteShader),
        SetGLFunctionPointer(DeleteTextures),
        SetGLFunctionPointer(DetachShader),
        SetGLFunctionPointer(DrawArrays),
        SetGLFunctionPointer(DrawBuffers),
        SetGLFunctionPointer(EnableVertexAttribArray),
        SetGLFunctionPointer(Finish),
        SetGLFunctionPointer(FramebufferTexture2D),
        SetGLFunctionPointer(GenBuffers),
        SetGLFunctionPointer(GenFramebuffers),
        SetGLFunctionPointer(GenTextures),
        SetGLFunctionPointer(GenVertexArrays),
        SetGLFunctionPointer(GetAttribLocation),
        SetGLFunctionPointer(GetError),
        SetGLFunctionPointer(GetIntegerv),
        SetGLFunctionPointer(GetProgramInfoLog),
        SetGLFunctionPointer(GetProgramiv),
        SetGLFunctionPointer(GetShaderInfoLog),
        SetGLFunctionPointer(GetShaderiv),
        SetGLFunctionPointer(GetString),
        SetGLFunctionPointer(GetUniformLocation),
        SetGLFunctionPointer(LinkProgram),
        SetGLFunctionPointer(ReadPixels),
        SetGLFunctionPointer(ShaderSource),
        SetGLFunctionPointer(TexImage2D),
        SetGLFunctionPointer(TexParameteri),
        SetGLFunctionPointer(TexSubImage2D),
        SetGLFunctionPointer(Uniform1f),
        SetGLFunctionPointer(Uniform1i),
        SetGLFunctionPointer(UseProgram),
        SetGLFunctionPointer(VertexAttribPointer),
        SetGLFunctionPointer(Viewport) {
    LOG(INFO) << "Constructed GLFunctionPointers";
  }

  void (*ActiveTexture)(GLenum texture);
  void (*AttachShader)(GLuint program, GLuint shader);
  void (*BindBuffer)(GLenum target, GLuint buffer);
  void (*BindFramebuffer)(GLenum target, GLuint framebuffer);
  void (*BindTexture)(GLenum target, GLuint texture);
  void (*BindVertexArray)(GLuint array);
  void (*BufferData)(GLenum target, GLsizeiptr size, const GLvoid *data,
                     GLenum usage);
  GLenum (*CheckFramebufferStatus)(GLenum target);
  void (*Clear)(GLbitfield mask);
  void (*CompileShader)(GLuint shader);
  GLuint (*CreateProgram)();
  GLuint (*CreateShader)(GLenum shader_type);
  void (*DeleteFramebuffers)(GLsizei n, const GLuint* framebuffers);
  void (*DeleteProgram)(GLuint program);
  void (*DeleteShader)(GLuint shader);
  void (*DeleteTextures)(GLsizei n, const GLuint* textures);
  void (*DetachShader)(GLuint program, GLuint shader);
  void (*DrawArrays)(GLenum mode, GLint first, GLsizei count);
  void (*DrawBuffers)(GLsizei n, const GLenum* bufs);
  void (*EnableVertexAttribArray)(GLuint index);
  void (*Finish)();
  void (*FramebufferTexture2D)(GLenum target, GLenum attachment,
                               GLenum textarget, GLuint texture, GLint level);
  void (*GenBuffers)(GLsizei n, GLuint* buffers);
  void (*GenFramebuffers)(GLsizei n, GLuint* ids);
  void (*GenTextures)(GLsizei n, GLuint* textures);
  void (*GenVertexArrays)(GLsizei n, GLuint* arrays);
  GLint (*GetAttribLocation)(GLuint program, const GLchar* name);
  GLenum (*GetError)();
  void (*GetIntegerv)(GLenum pname, GLint* data);
  void (*GetProgramInfoLog)(GLuint program, GLsizei maxLength, GLsizei* length,
                            GLchar* info_log);
  void (*GetProgramiv)(GLuint program, GLenum pname, GLint* params);
  void (*GetShaderInfoLog)(GLuint shader, GLsizei max_length, GLsizei* length,
                           GLchar* info_log);
  void (*GetShaderiv)(GLuint shader, GLenum pname, GLint* params);
  const GLubyte *(*GetString)(GLenum name);
  GLint (*GetUniformLocation)(GLuint program, const GLchar* name);
  void (*LinkProgram)(GLuint program);
  void (*ReadPixels)(GLint x, GLint y, GLsizei width, GLsizei height,
                     GLenum format, GLenum type, GLvoid* data);
  void (*ShaderSource)(GLuint shader, GLsizei count, const GLchar** string,
                       const GLint* length);
  void (*TexImage2D)(GLenum target, GLint level, GLint internal_format,
                     GLsizei width, GLsizei height, GLint border, GLenum format,
                     GLenum type, const GLvoid* data);
  void (*TexParameteri)(GLenum target, GLenum pname, GLint param);
  void (*TexSubImage2D)(GLenum target, GLint level, GLint xoffset,
                        GLint yoffset, GLsizei width, GLsizei height,
                        GLenum format, GLenum type, const GLvoid* data);
  void (*Uniform1f)(GLint location, GLfloat v0);
  void (*Uniform1i)(GLint location, GLint v0);
  void (*UseProgram)(GLuint program);
  void (*VertexAttribPointer)(GLuint index, GLint size, GLenum type,
                              GLboolean normalized, GLsizei stride,
                              const GLvoid* pointer);
  void (*Viewport)(GLint x, GLint y, GLsizei width, GLsizei height);
};

/*!
 * \brief Process global OpenGL workspace.
 */
class OpenGLWorkspace final : public DeviceAPI {
 public:
  ~OpenGLWorkspace() final;

  // override device API
  void SetDevice(TVMContext ctx) final;
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final;
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final;
  void* AllocWorkspace(TVMContext ctx, size_t size) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;

  /*!
   * \brief Get the global OpenGL workspace.
   * \return The global OpenGL workspace.
   */
  static const std::shared_ptr<OpenGLWorkspace>& Global();

  /*!
   * \brief Create an OpenGL program that uses the given fragment shader.
   * \param fragment_shader The fragment shader **source**.
   * \return The OpenGL program.
   */
  Program CreateProgram(const char* fragment_shader_src);

  /*!
   * \brief Create an OpenGL texture that stores an array.
   * \param type Element type.
   * \param nbytes Number of bytes in the array.
   * \return The OpenGL texture.
   */
  Texture CreateTexture(TVMType type, size_t nbytes);

  void PutTextureData(Texture* texture, GLint begin, GLsizei nelems,
                      const GLvoid* data);

  void GetTextureData(const Texture* texture, GLint begin, GLsizei nelems,
                      GLvoid* data);

  /*!
   * \brief Use an OpenGL program to render to a texture.
   * \param program The OpenGL program. Created by CreateProgram().
   * \param inputs All input textures.
   * \param output The output texture.
   */
  void Render(
      const Program& program,
      const std::vector<std::tuple<std::string, TVMType, void*>>& uniforms,
      const std::vector<std::pair<std::string, Texture*>>& inputs,
      Texture* output);

 private:
  friend class Texture;
  friend class Program;

  OpenGLWorkspace();

  GLFWwindow* window_;
  std::unique_ptr<GLFunctionPointers> gl;
  GLuint vertex_shader_;
  static const int kWindowWidth = 640;
  static const int kWindowHeight = 480;
  struct Vertex {
    float x, y;
  };
  static constexpr size_t kNumVertices = 6;
  static const Vertex vertices[kNumVertices];
  static const char* vertex_shader_text_;

  void BindTextureUnit(GLuint unit, GLuint texture);

  void OnDeleteTexture(GLuint texture);

  void OnDeleteProgram(GLuint program);

  /*!
   * \brief Check if there is any outstanding OpenGL error. If there is, crash.
   */
  void CheckOpenGLError();

  /*!
   * \brief Get the maximum number of texture units.
   */
  GLuint NumTextureUnits();

  /*!
   * \brief Create and compile a shader from a source string.
   * \param shader_kind The kind of shader.
   * Could be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
   * \param shader_src The source string of the shader.
   * \return The compiled shader ID.
   */
  GLuint CreateShader(GLenum shader_kind, const char* shader_src);

  /*!
   * \brief Create an OpenGL program that uses the given fragment shader.
   * \param fragment_shader The **compiled** fragment shader.
   * \return The OpenGL program.
   */
  Program CreateProgram(GLuint fragment_shader);
};

/*!
 * \brief An OpenGL program, composed of a vertex shader and a fragment shader.
 * In TVM, every program has the same vertex shader.
 * So a program just corresponds to a fragment shader.
 * A program can only be created by the workspace.
 * This class is just a wrapper over an OpenGL program ID.
 */
class Program {
 public:
  // Move constructor.
  Program(Program&& other) noexcept
      : workspace_(other.workspace_), program_(other.program_) {
    other.program_ = kInvalidProgram;
  }

  // Move assignment.
  Program& operator=(Program&& other) noexcept {
    workspace_ = other.workspace_;
    program_ = other.program_;
    other.program_ = kInvalidProgram;
    return *this;
  }

  // Disallow copy.
  Program(const Program& other) = delete;
  Program& operator=(const Program& other) = delete;

  // Destructor.
  ~Program() {
    if (program_ != kInvalidProgram) {
      workspace_->OnDeleteProgram(program_);
      program_ = kInvalidProgram;
    }
  }

 private:
  friend class OpenGLWorkspace;

  // Only OpenGLWorkspace can create a Program.
  // We enforce this to make sure OpenGL is initialized.
  explicit Program(OpenGLWorkspace* workspace, GLuint program)
      : workspace_(workspace), program_(program) {}

  // The internal OpenGL program ID.
  GLuint program() { return program_; }

  static constexpr GLuint kInvalidProgram = static_cast<GLuint>(-1);

  OpenGLWorkspace* workspace_;
  GLuint program_;
};

struct TextureFormat {
  TextureFormat(GLint internal_format, GLenum format, GLenum type)
      : internal_format(internal_format), format(format), type(type) {}

  GLsizei elemsz() const {
    switch (type) {
      case GL_BYTE: case GL_UNSIGNED_BYTE:
        return 1;
      case GL_SHORT: case GL_UNSIGNED_SHORT:
        return 2;
      case GL_INT: case GL_UNSIGNED_INT:
        return 4;
      case GL_FLOAT:
        return 4;
      default:
        LOG(FATAL) << "Unsupported type";
        return -1;
    }
  }

  bool operator==(const TextureFormat& other) const {
    return std::make_tuple(internal_format, format, type) ==
        std::make_tuple(other.internal_format, other.format, other.type);
  }

  GLint internal_format;  // OpenGL says this is GLint, not GLenum.
  GLenum format;
  GLenum type;
};

/*!
 * \brief An OpenGL texture represents a chunk of GPU memory.
 * This is the way we represent tensors.
 * We always use 2D textures.
 */
class Texture {
 public:
  // Move constructor.
  Texture(Texture&& other) noexcept
      : workspace_(other.workspace_), texture_(other.texture_),
        format_(other.format_), width_(other.width_), height_(other.height_) {
    other.texture_ = kInvalidTexture;
  }

  // Move assignment.
  Texture& operator=(Texture&& other) noexcept {
    workspace_ = other.workspace_;
    texture_ = other.texture_;
    format_ = other.format_;
    width_ = other.width_;
    height_ = other.height_;
    other.texture_ = kInvalidTexture;
    return *this;
  }

  // Disallow copy.
  Texture(const Texture& other) = delete;
  Texture& operator=(const Texture& other) = delete;

  // Destructor.
  ~Texture() {
    if (texture_ != kInvalidTexture) {
      LOG(INFO) << "Deleting texture [" << texture_ << "]";
      workspace_->OnDeleteTexture(texture_);
      texture_ = kInvalidTexture;
    }
  }

  /*!
   * \brief The width of the texture in number of pixels.
   */
  GLsizei width() const { return width_; }

  /*!
   * \brief The height of the texture in number of pixels.
   */
  GLsizei height() const { return height_; }

  /*!
   * \brief The number of bytes of each element in the array.
   */
  GLsizei elemsz() const { return format_.elemsz(); }

 private:
  friend class OpenGLWorkspace;

  // Only OpenGLWorkspace can create a Texture.
  // We enforce this to make sure OpenGL is initialized.
  // Always only use the first dimension of a 2D texture.
  // The reason is that texelFetch only supports 2D textures.
  explicit Texture(OpenGLWorkspace* workspace, GLuint texture,
                   TextureFormat format,
                   GLsizei width, GLsizei height)
      : workspace_(workspace), texture_(texture), format_(format),
        width_(width), height_(height) {}

  // The internal texture ID.
  GLuint texture() const { return texture_; }

  static constexpr GLuint kInvalidTexture = static_cast<GLuint>(-1);

  OpenGLWorkspace* workspace_;
  GLuint texture_;
  TextureFormat format_;
  GLsizei width_;
  GLsizei height_;
};

}  // namespace gl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_
