if(USE_ANTLR)
  if(EXISTS /usr/local/lib/antlr-4.7.1-complete.jar)
    set(ANTLR4 "/usr/local/lib/antlr-4.7.1-complete.jar")

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
      COMMAND $ENV{JAVA_HOME}/bin/java -jar ${ANTLR4} -visitor -no-listener -Dlanguage=Python2 ${RELAY_PARSER_DIR}/Relay.g4 -o ${RELAY_PARSER_DIR}/py2
      COMMAND $ENV{JAVA_HOME}/bin/java -jar ${ANTLR4} -visitor -no-listener -Dlanguage=Python3 ${RELAY_PARSER_DIR}/Relay.g4 -o ${RELAY_PARSER_DIR}/py3
      DEPENDS ${RELAY_PARSER_DIR}/Relay.g4
      WORKING_DIRECTORY ${RELAY_PARSER_DIR})

    add_custom_target(relay_parser ALL DEPENDS ${RELAY_PARSER})
  else()
    message(FATAL_ERROR "Can't find ANTLR4!")
  endif()
endif(USE_ANTLR)
