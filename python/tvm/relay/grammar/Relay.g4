/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// TODO: We need some way of indicating to users that you need to enable
// USE_ANTLR in config.cmake.
/*
 * NOTE: All upper-case rules are *lexer* rules and all lower-case rules are *parser* rules.
 */

grammar Relay;

SEMVER: 'v0.0.3' ;

// Lexing
// comments
COMMENT : '/*' (COMMENT|.)*? '*/' -> skip;
WS : [ \t\n\r]+ -> skip;
LINE_COMMENT : '//' .*? '\n' -> skip;

fragment ESCAPED_QUOTE : '\\"';
QUOTED_STRING :   '"' ( ESCAPED_QUOTE | ~('\n'|'\r') )*? '"';

// operators
MUL: '*' ;
DIV: '/' ;
ADD: '+' ;
SUB: '-' ;
LT: '<' ;
GT: '>' ;
LE: '<=' ;
GE: '>=' ;
EQ: '==' ;
NE: '!=' ;

BOOL_LIT
  : 'True'
  | 'False'
  ;

START_UPPER_CNAME: UPPER_LETTER ('_'|LETTER|DIGIT)*;
START_LOWER_CNAME: LOWER_LETTER ('_'|LETTER|DIGIT)*;
// CNAME: ('_'|LETTER) ('_'|LETTER|DIGIT)* ('.' CNAME)* ;

// non-negative floats
fragment PREFLOAT : NAT ('.' NAT)? EXP?; // 1.35, 1.35E-9, 0.3, 4.5, 1, 1e10 3e4

FLOAT : PREFLOAT 'f';

// BASE_TYPE : ('int'|'uint'|'float'|'bool') DIGIT*;
baseType : 'int32' ;

// non-negative ints
NAT: DIGIT+ ;
fragment EXP: [eE] [+\-]? NAT ; // \- since - means "range" inside [...]

fragment LOWER_LETTER: [a-z];
fragment UPPER_LETTER: [A-Z];
fragment LETTER: [a-zA-Z];
fragment DIGIT: [0-9];

METADATA: 'METADATA:' .*;
// Parsing

// A Relay program is a list of global definitions or an expression.
// prog: SEMVER (defn+ | expr) METADATA? EOF ;
prog: SEMVER (defn* | expr) METADATA? EOF ;

opIdent: START_LOWER_CNAME ('.' START_LOWER_CNAME)*;
globalVar: '@' START_LOWER_CNAME ;
localVar: '%' START_LOWER_CNAME ;
// TODO(weberlo): why does 'int32` generate a parse error when it's literally a
// lexer token?
// typeIdent: BASE_TYPE | START_UPPER_NAME ;
typeIdent: (baseType | START_UPPER_CNAME) ;
graphVar: '%' NAT ;

exprList: (expr (',' expr)*)?;
callList
  : exprList            # callNoAttr
  | (expr ',')* attrSeq # callWithAttr
  ;

expr
  // operators
  : '(' expr ')'                              # paren
  | '{' expr '}'                              # paren
  // function application
  | expr '(' callList ')'                     # call
  | '-' expr                                  # neg
  | expr op=('*'|'/') expr                    # binOp
  | expr op=('+'|'-') expr                    # binOp
  | expr op=('<'|'>'|'<='|'>=') expr          # binOp
  | expr op=('=='|'!=') expr                  # binOp
  // function definition
  | func                                      # funcExpr
  // tuples and tensors
  | '(' ')'                                   # tuple
  | '(' expr ',' ')'                          # tuple
  | '(' expr (',' expr)+ ')'                  # tuple
  | '[' (expr (',' expr)*)? ']'               # tensor
  | 'if' '(' expr ')' body 'else' body        # ifElse
  | matchType '(' expr ')' '{' matchClause+ '}'    # match
  | expr '.' NAT                              # projection
  // sequencing
  | 'let' var '=' expr ';' expr               # let
  // sugar for let %_ = expr; expr
  | expr ';;' expr                            # let
  | graphVar '=' expr ';' expr                # graph
  | ident                                     # identExpr
  | scalar                                    # scalarExpr
  | meta                                      # metaExpr
  | QUOTED_STRING                             # stringExpr
  ;

func: 'fn'        typeParamList? '(' argList ')' ('->' typeExpr)? body ;
defn
  : 'def' globalVar typeParamList? '(' argList ')' ('->' typeExpr)? body  # funcDefn
  | 'type' typeIdent typeParamList? '=' adtConstructor+                   # adtDefn
  ;

adtConstructor: '|' constructorName ('(' typeExpr (',' typeExpr)* ')')? ;
matchClause: '|' constructorName patternList? '=>' expr ;
matchType : 'match' | 'match?' ;

// TODO: Will need to make this recursive
patternList: '(' pattern (',' pattern)* ')';
pattern
  : '_'
  | localVar (':' typeExpr)?
  ;

// constructorName: typeIdent ;
constructorName: START_UPPER_CNAME ;

argList
  : varList              # argNoAttr
  | (var ',')* attrSeq   # argWithAttr
  ;

varList: (var (',' var)*)?;
var: localVar (':' typeExpr)?;

attrSeq: attr (',' attr)*;
// attr: LOWER_NAME '=' expr ;
attr: START_LOWER_CNAME '=' expr ;

typeExpr
  : '(' ')'                                                      # tupleType
  | '(' typeExpr ',' ')'                                            # tupleType
  | '(' typeExpr (',' typeExpr)+ ')'                                   # tupleType
  | typeIdent typeParamList                                     # typeCallType
  | typeIdent                                                    # typeIdentType
  | 'Tensor' '[' shapeList ',' typeExpr ']'                         # tensorType
  | 'fn' typeParamList? '(' (typeExpr (',' typeExpr)*)? ')' '->' typeExpr # funcType
  // | '_'                                                          # incompleteType
  // TODO: Why the fuck does this rule exist?
  // | NAT                                                          # intType
  ;

// TODO: For some reason, spaces aren't allowed between type params?
typeParamList: '[' typeIdent (',' typeIdent)* ']' ;

shapeList
  : '(' shape (',' shape)+ ')'
  | '(' ')'
  | shape
  ;

// meta : 'meta' '[' LOWER_NAME ']' '[' NAT ']';
meta : 'meta' '[' START_LOWER_CNAME ']' '[' NAT ']';

shape
  : meta # metaShape
  | '(' shape ')'                        # parensShape
  | NAT                                  # intShape
  ;

// int8, int16, int32, int64
// uint8, uint16, uint32, uint64
// float16, float32, float64
// bool

body: '{' expr '}' ;

scalar
  : FLOAT    # scalarFloat
  | NAT      # scalarInt
  | BOOL_LIT # scalarBool
  ;

ident
  : opIdent
  | globalVar
  | localVar
  | typeExpr
  | graphVar
  ;
