#include <cstdio>
#include <iostream>

#include "sgx_urts.h"
#include "sgx_eid.h"
#include "test_addone_u.h"
#include "../../sgx/runtime_u.cc"

#define TOKEN_FILENAME   "bin/test_addone.token"
#define ENCLAVE_FILENAME "lib/test_addone.signed.so"

sgx_enclave_id_t tvm_sgx_eid;

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
  sgx_status = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &tvm_sgx_eid, NULL);
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
  if(initialize_enclave() < 0) {
    printf("Failed to initialize enclave.\n");
    return -1;
  }

  /* Run TVM within the enclave */
  int addone_status;
  sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
  sgx_status = tvm_ecall_run_module(tvm_sgx_eid, nullptr, &addone_status);
  if (sgx_status != SGX_SUCCESS) {
    print_error_message(sgx_status);
  }

  sgx_destroy_enclave(tvm_sgx_eid);

  if (addone_status == 1) {
    printf("It works!");
    return 0;
  }
  printf("It doesn't work.");
  return -1;
}

extern "C" {
void ocall_println(const char* str) {
  std::cout << "Enclave says: " << str << std::endl;
}
}
