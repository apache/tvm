add_subdirectory(ir)
add_subdirectory(transform)

# All mlir relay libraries.
set(OMLibs
        OMRelayOps
        OMhloOps
        RelayMlirTransform
        RelayMlirTranslate
        RelayMlirTranslateRegistration
        MhloUtils)

set(OMLibs ${OMLibs} PARENT_SCOPE)

add_subdirectory(tool)
add_subdirectory(translate)