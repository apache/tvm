# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

include(ExternalProject)

# Check if xcodebuild tool is available and configured.
# Otherwise will skip all iOS specific targets.
execute_process(COMMAND xcodebuild -version
        RESULT_VARIABLE XCBUILD_AVAILABLE
        OUTPUT_QUIET
        ERROR_QUIET
        )

if (NOT XCBUILD_AVAILABLE EQUAL 0)
    message(WARNING
            "The build tool xcodebuild is not properly configured. Please install Xcode app and specify "
            "path to it via DEVELOPER_DIR env var or \"sudo xcode-select -switch <path-to-xcode-dev-dir>\".\n"
            "iOS RPC application target is switched off."
            )
    return()
endif()

# External project with custom mach-o dynamic loader
# It is required to load unsigned shared modules on real iOS devices
ExternalProject_Add(custom_dso_loader
        GIT_REPOSITORY https://github.com/octoml/macho-dyld.git
        GIT_TAG d1f7032e7882bc060b49a4fb058f50a23668b074
        PREFIX custom_dso_loader
        LOG_DOWNLOAD TRUE
        LOG_CONFIGURE TRUE
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>  # to install into local build dir
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
            -DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=${CMAKE_BUILD_WITH_INSTALL_NAME_DIR}
        )

if(NOT CMAKE_IOS_RPC_BUNDLE)
    set(CMAKE_IOS_RPC_BUNDLE org.apache.tvmrpc)
endif()

# iOS RPC Xcode project wrapper to integrate into Cmake
ExternalProject_Add(ios_rpc
        PREFIX ios_rpc
        DEPENDS custom_dso_loader tvm_runtime
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
        CONFIGURE_COMMAND ""
        INSTALL_COMMAND ""
        BUILD_COMMAND xcodebuild
            -target tvmrpc
            -configuration ${CMAKE_BUILD_TYPE}
            -project <SOURCE_DIR>/tvmrpc.xcodeproj
            -sdk ${CMAKE_OSX_SYSROOT}
            -arch ${CMAKE_OSX_ARCHITECTURES}
            -hideShellScriptEnvironment
            -allowProvisioningUpdates
            build
                SYMROOT=<BINARY_DIR>
                IPHONEOS_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
                DEVELOPMENT_TEAM=${CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM}
                TVM_BUILD_DIR=${CMAKE_BINARY_DIR}
                USE_CUSTOM_DSO_LOADER=1
                PRODUCT_BUNDLE_IDENTIFIER=${CMAKE_IOS_RPC_BUNDLE}
        )
