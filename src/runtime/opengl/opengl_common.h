/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_common.h
 * \brief OpenGL common header
 */
#ifndef TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_
#define TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_

#include <mutex>
#include <tvm/runtime/config.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <dmlc/logging.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace tvm {
namespace runtime {
namespace gl {
    class Texture;
    class Program;
/*!
 * \brief Process global OpenGL workspace.
 */
class OpenGLWorkspace final : public DeviceAPI {
 public:
  // whether the workspace it initialized.
  bool initialized_{false};
  // the mutex for initialization
  std::mutex mu;
  void Init();
    ~OpenGLWorkspace();

  // override device API
  void SetDevice(TVMContext ctx) final;
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) final;
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
    std::shared_ptr<Program> CreateProgram(const char *fragment_shader_src);
  // get the global workspace
  static const std::shared_ptr<OpenGLWorkspace>& Global();
    void Render(
            const Program &program,
            const std::vector<std::pair<std::string, Texture*>> &inputs,
            Texture* output);

private:
    friend class Texture;
    GLFWwindow *window_;
    GLuint vertex_shader_;
    static const int kWindowWidth = 640;
    static const int kWindowHeight = 480;
    struct Vertex {
        float x, y;
    };
    static constexpr size_t kNumVertices = 6;
    static const Vertex vertices[kNumVertices];
    static const char *vertex_shader_text_;
    void BindTextureUnit(GLuint unit, GLuint texture);
    GLuint NumTextureUnits();
    GLuint CreateShader(GLenum shader_kind, const char *shader_src);
    std::shared_ptr<Program> CreateProgram(GLuint fragment_shader);
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
        Program() : program_(kInvalidProgram) {};
        explicit Program(GLuint program) : program_(program) {};
        // Move constructor.
        Program(Program &&other) noexcept : program_(other.program_) {
          other.program_ = kInvalidProgram;
        }
        // Cannot be copied.
        Program(const Program &other) = delete;
        // Cannot be assigned.
        Program &operator=(const Program &other) = delete;
        ~Program();
    private:
        friend class OpenGLWorkspace;
        // The internal OpenGL program ID.
        GLuint program_;
        static const GLuint kInvalidProgram = static_cast<GLuint>(-1);
    };

    /*!
         * An OpenGL texture represents a chunk of GPU memory.
         * This is the way we represent tensors.
         * We always use 2D textures.
         */
    class Texture {
    public:
        explicit Texture(size_t size);
        ~Texture();
        Texture(Texture &&other) noexcept
                : texture_(other.texture_), width_(other.width_), height_(other.height_) {
            other.texture_ = kInvalidTexture;
        }
        Texture(const Texture &other) = delete;
        Texture &operator=(const Texture &other) = delete;
        GLsizei width() const { return width_; }
        GLsizei height() const { return height_; }
        void GetData(GLfloat *data) const;
        void PutData(size_t size, const void *data);
    private:
        friend class OpenGLWorkspace;
        GLuint texture() const { return texture_; }
        static const GLuint kInvalidTexture = static_cast<GLuint>(-1);
        GLuint texture_;
        GLsizei width_;
        GLsizei height_;
    };

}  // namespace gl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_
