#include <cstdio>
#include <sstream>
#include <fstream>
#include <iostream>

#include "sgx_urts.h"
#include "sgx_eid.h"
#include "model_u.h"

#define TOKEN_FILENAME   "bin/enclave.token"
#define ENCLAVE_FILENAME "lib/model.signed.so"

sgx_enclave_id_t global_eid = 0;  // global EID shared by multiple threads

typedef struct _sgx_errlist_t {
  sgx_status_t err;
  const char *msg;
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
  { SGX_ERROR_DEVICE_BUSY, "SGX device was busy." },
  { SGX_ERROR_ENCLAVE_FILE_ACCESS, "Can't open enclave file." },
  { SGX_ERROR_ENCLAVE_LOST, "Power transition occurred." },
  { SGX_ERROR_INVALID_ATTRIBUTE, "Enclave was not authorized." },
  { SGX_ERROR_INVALID_ENCLAVE, "Invalid enclave image." },
  { SGX_ERROR_INVALID_ENCLAVE_ID, "Invalid enclave identification." },
  { SGX_ERROR_INVALID_METADATA, "Invalid enclave metadata." },
  { SGX_ERROR_INVALID_PARAMETER, "Invalid parameter." },
  { SGX_ERROR_INVALID_SIGNATURE, "Invalid enclave signature." },
  { SGX_ERROR_INVALID_VERSION, "Enclave version was invalid." },
  { SGX_ERROR_MEMORY_MAP_CONFLICT, "Memory map conflicted." },
  { SGX_ERROR_NO_DEVICE, "Invalid SGX device." },
  { SGX_ERROR_OUT_OF_EPC, "Out of EPC memory." },
  { SGX_ERROR_OUT_OF_MEMORY, "Out of memory." },
  { SGX_ERROR_UNEXPECTED, "Unexpected error occurred." },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t status)
{
  size_t idx = 0;
  size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

  for (idx = 0; idx < ttl; idx++) {
    if(status == sgx_errlist[idx].err) {
      printf("Error: %s\n", sgx_errlist[idx].msg);
      break;
    }
  }

  if (idx == ttl)
    printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", status);
}

/* Initialize the enclave:
 *   Step 1: try to retrieve the launch token saved by last transaction
 *   Step 2: call sgx_create_enclave to initialize an enclave instance
 *   Step 3: save the launch token if it is updated
 */
int initialize_enclave(void)
{
  sgx_launch_token_t token = {0};
  sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
  int updated = 0;

  /* Step 1: try to retrieve the launch token saved by last transaction
   *     if there is no token, then create a new one.
   */
  FILE *fp = fopen(TOKEN_FILENAME, "rb");
  if (fp == NULL && (fp = fopen(TOKEN_FILENAME, "wb")) == NULL) {
    printf("Warning: Failed to create/open the launch token file \"%s\".\n", TOKEN_FILENAME);
    return -1;
  }

  /* read the token from saved file */
  size_t read_num = fread(token, 1, sizeof(sgx_launch_token_t), fp);
  if (read_num != 0 && read_num != sizeof(sgx_launch_token_t)) {
    /* if token is invalid, clear the buffer */
    memset(&token, 0x0, sizeof(sgx_launch_token_t));
    printf("Warning: Invalid launch token read from \"%s\".\n", TOKEN_FILENAME);
  }

  /* Step 2: call sgx_create_enclave to initialize an enclave instance */
  /* Debug Support: set 2nd parameter to 1 */
  sgx_status = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &global_eid, NULL);
  if (sgx_status != SGX_SUCCESS) {
    print_error_message(sgx_status);
    if (fp != NULL) fclose(fp);
    return -1;
  }

  /* Step 3: save the launch token if it is updated */
  if (updated == 0 || fp == NULL) {
    /* if the token is not updated, or file handler is invalid, do not perform saving */
    if (fp != NULL) fclose(fp);
    return 0;
  }

  /* reopen the file with write capablity */
  fp = freopen(TOKEN_FILENAME, "wb", fp);
  if (fp == NULL) return 0;
  size_t write_num = fwrite(token, 1, sizeof(sgx_launch_token_t), fp);
  if (write_num != sizeof(sgx_launch_token_t))
    printf("Warning: Failed to save launch token to \"%s\".\n", TOKEN_FILENAME);
  fclose(fp);
  return 0;
}

int SGX_CDECL main(int argc, char *argv[]) {
  if(initialize_enclave() < 0){
    printf("Failed to initialize enclave.\n");
    return -1;
  }

  std::ifstream f_img("bin/cat.bin", std::ios::binary);
  std::string img(static_cast<std::stringstream const&>(
                  std::stringstream() << f_img.rdbuf()).str());

  unsigned predicted_class;
  sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
  sgx_status = ecall_infer(global_eid, &predicted_class, img.c_str());
  if (sgx_status != SGX_SUCCESS) {
    print_error_message(sgx_status);
  }

  sgx_destroy_enclave(global_eid);
  if (predicted_class == 281) {
    std::cout << "It's a tabby!" << std::endl;
    return 0;
  }
  std::cerr << "Inference failed! Predicted class: " <<
    predicted_class << std::endl;
  return 1;
}
