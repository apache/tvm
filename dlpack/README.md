# DLPack: Open In Memory Tensor Structure

[![Build Status](https://travis-ci.org/dmlc/dlpack.svg?branch=master)](https://travis-ci.org/dmlc/dlpack)

DLPack is an open in-memory tensor structure to for sharing tensor among frameworks. DLPack enables

- Easier sharing of operators between deep learning frameworks.
- Easier wrapping of vendor level operator implementations, allowing collaboration when introducing new devices/ops.
- Quick swapping of backend implementations, like different version of BLAS
- For final users, this could bring more operators, and possiblity of mixing usage between frameworks.

We do not intend to implement of Tensor and Ops, but instead use this as common bridge
to reuse tensor and ops across frameworks.

## Proposal Procedure
RFC proposals are opened as issues. The major release will happen as a vote issue to make
sure the participants agree on the changes.

## Project Structure
There are two major components so far
- include: stablized headers
- contrib: in progress unstable libraries

## People
Here are list of people who have been involved in DLPack RFC design proposals:

@soumith @piiswrong @Yangqing @naibaf7 @bhack @edgarriba @tqchen @prigoyal @zdevito
