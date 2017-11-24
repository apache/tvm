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

    inline const char *GLGetErrorString(GLenum error) {
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
    // TODO(zhixunt): When porting to TVM, change this to
//   CHECK(err == GL_NO_ERROR) << ...;
    void OPENGL_CHECK_ERROR() {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "OpenGL error, code=" << err << ": "
                      << gl::GLGetErrorString(err) << std::endl;
            assert(false);
        }
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

        void GlfwErrorCallback(int err, const char *str) {
            std::cerr << "Error: [" << err << "] " << str << std::endl;
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
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue *rv) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::GetAttr";
}

void* OpenGLWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream()
      << "OpenGLWorkspace::AllocDataSpace(ctx, size = "
      << size << ", alignment = " << alignment << ")";
    this->Init();
// Create a texture.
    GLuint texture;
    OPENGL_CALL(glGenTextures(1, &texture));

    LOG_INFO.stream() << "Created texture [" << texture << "]";

  return reinterpret_cast<void*>(texture);
}

void OpenGLWorkspace::FreeDataSpace(TVMContext ctx, void *ptr) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream()
      << "OpenGLWorkspace::FreeDataSpace(ctx, ptr = "
      << ptr << ")";
}

void OpenGLWorkspace::CopyDataFromTo(const void *from,
                                     size_t from_offset,
                                     void *to,
                                     size_t to_offset,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMStreamHandle stream) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream()
      << "OpenGLWorkspace::CopyDataFromTo("
      << "from = " << from << ", "
      << "from_offset = " << from_offset << ", "
      << "to = " << to << ", "
      << "to_offset = " << to_offset << ", "
      << "size = " << size << ", "
      << "ctx_from = (" << ctx_from.device_type << ", " << ctx_from.device_id << "), "
      << "ctx_to = (" << ctx_to.device_type << ", " << ctx_to.device_id << "), stream)";
    this->Init();
    CHECK(stream == nullptr);
    if (ctx_from.device_type == kDLOpenGL && ctx_to.device_type == kDLOpenGL) {
    } else if (ctx_from.device_type == kDLOpenGL && ctx_to.device_type == kDLCPU) {
        auto texture = (GLuint) reinterpret_cast<uintptr_t>(from);
        BindTextureUnit(NumTextureUnits() - 1, texture);

        OPENGL_CALL(glGetTexImage(GL_TEXTURE_2D, /*level=*/0, GL_RED, GL_FLOAT, to));
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLOpenGL) {
        auto texture = (GLuint) reinterpret_cast<uintptr_t>(to);
        // Bind to temporary unit.
        BindTextureUnit(NumTextureUnits() - 1, texture);

        // Similar to cudaMemcpy.
        // TODO(pengw): How can we know the type of data?
        OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_R32F,
                                 (GLsizei)size / sizeof(GLfloat), 1, /*border=*/0,
                                 GL_RED, GL_FLOAT, from));

        // TODO(zhixunt): What are these?
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        OPENGL_CALL(
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
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

void OpenGLWorkspace::FreeWorkspace(TVMContext ctx, void *data) {
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
    std::cout << "glfwInit() failed!" << std::endl;
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
    std::cout << "glfwCreateWindow() failed!" << std::endl;
    assert(false);
  }

  std::cout << "GLFW says OpenGL version: "
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MAJOR)
            << "."
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MINOR)
            << "."
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_REVISION)
            << std::endl;

  // Before using any OpenGL API, we must specify a context.
  glfwMakeContextCurrent(window_);

  // Must be called after creating GLFW window.
  gladLoadGL();

  std::cout << "Opengl says version: " << glGetString(GL_VERSION) << std::endl;

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

TVM_REGISTER_GLOBAL("device_api.opengl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = OpenGLWorkspace::Global().get();
    *rv = static_cast<void*>(ptr);
  });

        GLuint OpenGLWorkspace::CreateShader(GLenum shader_kind, const char *shader_src) {
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
                std::cout << err_msg.get() << std::endl;
                assert(false);
            }

            OPENGL_CHECK_ERROR();

            return shader;
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
        const char *OpenGLWorkspace::vertex_shader_text_ = "#version 330 core\n"
                "in vec2 point; // input to vertex shader\n"
                "void main() {\n"
                "  gl_Position = vec4(point, 0.0, 1.0);\n"
                "}\n";

}  // namespace gl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENGL_RUNTIME
