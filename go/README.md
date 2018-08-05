# gotvm - Golang Frontend for TVM Runtime

This folder contain golang interface for TVM runtime. It brings TVM runtime to Golang.

- It enable c runtime api of tvm exposed to golang.
- It enables module loading (lib, graph and params) and inference operations.

## Installation

### Requirements

- go compiler (https://golang.org/)

### Modules

- runtime
  Module that generates golang package corresponding to the c runtime api exposed from tvm source tree.
  This process build golang package _gotvm.a_

- sample
  Sample golang reference applications to inference through gotvm package.

### Build

Once the Requirements are installed

```bash
make
```

Compilation process builds go package in runtime module and a sample applications.
  simple : golang application for a module build by sample/deploy.py _(C = A + B)_
  complex : golang application to load module library, graph and params on graph runtime and run.

## Run

simple : Use the deploy.so build by make.
complex : Reference application to deploy a realtime module with lib, graph and param.

## TODO

Function registration API are not implemented.

## Documentation
gotvm.go is very well documented with sufficient information about gotvm package.
A html version documentation can be accessed by running below command after building runtime.

```bash
godoc -http=:6060  -goroot=./gopath
```
After above command try http://127.0.0.1:6060 from any browser.
