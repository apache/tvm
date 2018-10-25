find_program(ANTLR4 antlr4)

if(USE_ANTLR)
  find_program(ANTLR4 antlr4)
  if(NOT ANTLR4)
    message(FATAL_ERROR "Can't find ANTLR4!")
  endif()

  set(RELAY_PARSER_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm/relay/grammar)

  set(RELAY_PARSER
    ${RELAY_PARSER_DIR}/py2/RelayVisitor.py
    ${RELAY_PARSER_DIR}/py2/RelayParser.py
    ${RELAY_PARSER_DIR}/py2/RelayLexer.py

    ${RELAY_PARSER_DIR}/py3/RelayVisitor.py
    ${RELAY_PARSER_DIR}/py3/RelayParser.py
    ${RELAY_PARSER_DIR}/py3/RelayLexer.py)

  # Generate ANTLR grammar for parsing.
  add_custom_command(OUTPUT ${RELAY_PARSER}
    COMMAND antlr4 -visitor -no-listener -Dlanguage=Python2 ${RELAY_PARSER_DIR}/Relay.g4 -o ${RELAY_PARSER_DIR}/py2
    COMMAND antlr4 -visitor -no-listener -Dlanguage=Python3 ${RELAY_PARSER_DIR}/Relay.g4 -o ${RELAY_PARSER_DIR}/py3
    DEPENDS ${RELAY_PARSER_DIR}/Relay.g4
    WORKING_DIRECTORY ${RELAY_PARSER_DIR})

  add_custom_target(relay_parser ALL DEPENDS ${RELAY_PARSER})
endif(USE_ANTLR)
