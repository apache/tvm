// attr [A] storage_scope = "local.L0A"
allocate A[int16 * 600]
for (i, 0, n) {
  for (j, 0, 10) {
    A[j] = (int16)1
  }
  for (j, 0, 20) {
    A[j] = (int16)1
  }
  for (j, 0, 10) {
    A[(100 + j)] = 1.200000f
  }
}
// attr [A] storage_scope = "local.L0A"
allocate A[int16 * 800]
for (i, 0, n) {
  for (j, 0, 10) {
    A[j] = (int16)1
  }
  for (j, 0, 20) {
    A[(200 + j)] = (int16)1
  }
  for (j, 0, 20) {
    A[j] = A[j]
  }
  for (j, 0, 10) {
    A[(200 + j)] = 1.200000f
  }
}
// attr [A] storage_scope = "local.L0A"
allocate A[int16 * 800]
for (i, 0, n) {
  for (j, 0, 10) {
    A[j] = (int16)1
  }
  for (j, 0, 20) {
    A[(200 + j)] = (int16)1
  }
  for (j, 0, 20) {
    A[j] = A[j]
  }
  for (j, 0, 10) {
    A[(200 + j)] = 1.200000f
  }
}
