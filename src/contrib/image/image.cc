/*!
 *  Copyright (c) 2018 by Contributors
 * \file Use external image library (jpeg, png, bml, gif ...etc) call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>

extern "C" {
#include <jpeglib.h>
#include <setjmp.h>

struct jpeg_internal_source_mgr {
  struct jpeg_source_mgr pub;
  const JOCTET *data;
  size_t       len;
};

static const JOCTET EOI_BUFFER[1] = { JPEG_EOI };

static void jpeg_internal_init_source(j_decompress_ptr cinfo) {}

static boolean jpeg_internal_fill_input_buffer(j_decompress_ptr cinfo) {
  jpeg_internal_source_mgr* src = reinterpret_cast<jpeg_internal_source_mgr*>(cinfo->src);
  src->pub.next_input_byte = EOI_BUFFER;
  src->pub.bytes_in_buffer = 1;
  return TRUE;
}

static void jpeg_internal_skip_input_data(j_decompress_ptr cinfo, int64_t num_bytes) {
  jpeg_internal_source_mgr* src = reinterpret_cast<jpeg_internal_source_mgr*>(cinfo->src);
  if (src->pub.bytes_in_buffer < (unsigned)num_bytes) {
    src->pub.next_input_byte = EOI_BUFFER;
    src->pub.bytes_in_buffer = 1;
  } else {
    src->pub.next_input_byte += num_bytes;
    src->pub.bytes_in_buffer -= num_bytes;
  }
}

static void jpeg_internal_term_source(j_decompress_ptr cinfo) {}

static void jpeg_internal_set_source_mgr(j_decompress_ptr cinfo, const char* data, size_t len) {
  jpeg_internal_source_mgr* src;
  if (cinfo->src == 0) {
    cinfo->src = (struct jpeg_source_mgr *)(*cinfo->mem->alloc_small)
                 ((j_common_ptr) cinfo, JPOOL_PERMANENT, sizeof(jpeg_internal_source_mgr));
  }
  src = reinterpret_cast<jpeg_internal_source_mgr*>(cinfo->src);
  src->pub.init_source = jpeg_internal_init_source;
  src->pub.fill_input_buffer = jpeg_internal_fill_input_buffer;
  src->pub.skip_input_data = jpeg_internal_skip_input_data;
  src->pub.resync_to_restart = jpeg_resync_to_restart;
  src->pub.term_source = jpeg_internal_term_source;
  src->data = (const JOCTET *)data;
  src->len = len;
  src->pub.bytes_in_buffer = len;
  src->pub.next_input_byte = src->data;
}

}

namespace tvm {
namespace contrib {

using namespace runtime;

// image decoder for types jpeg, gif, png, bmp ..etc
TVM_REGISTER_GLOBAL("tvm.contrib.image.decode")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    DLTensor* image_input  = args[0];
    DLTensor* image_output = args[1];

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    int row_stride;                     /* physical row width in output buffer */

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_internal_set_source_mgr(&cinfo, (static_cast<char*>(image_input->data)
                                          + image_input->byte_offset), image_input->shape[0]);

    (void) jpeg_read_header(&cinfo, TRUE);

    CHECK_EQ(cinfo.image_height, image_output->shape[0]);
    CHECK_EQ(cinfo.image_width, image_output->shape[1]);
    CHECK_EQ(cinfo.num_components, image_output->shape[2]);

    (void) jpeg_start_decompress(&cinfo);
    /* JSAMPLEs per row in output buffer */
    row_stride = cinfo.output_width * cinfo.output_components;

    unsigned char *buffer[1];

    for (int ii=0 ; cinfo.output_scanline < cinfo.output_height; ii++) {
        buffer[0] = (static_cast<unsigned char*>(image_output->data)
                    + image_output->byte_offset) + (cinfo.output_scanline) * row_stride;
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);
    }

    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
  });
}  // namespace contrib
}  // namespace tvm
