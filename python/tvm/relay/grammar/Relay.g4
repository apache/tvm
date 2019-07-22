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

// list = *, seq = ?

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

CNAME: ('_'|LETTER) ('_'|LETTER|DIGIT)* ('.' CNAME)*;
opIdent: CNAME ;
GLOBAL_VAR: '@' CNAME ;
LOCAL_VAR: '%' CNAME;
GRAPH_VAR: '%' NAT;

DATATYPE : 'int64';
// non-negative floats
fragment PREFLOAT : NAT ('.' NAT)? EXP?; // 1.35, 1.35E-9, 0.3, 4.5, 1, 1e10 3e4

FLOAT : PREFLOAT 'f';

// non-negative ints
NAT: DIGIT+ ;
fragment EXP: [eE] [+\-]? NAT ; // \- since - means "range" inside [...]

fragment LETTER: [a-zA-Z];
fragment DIGIT: [0-9];

METADATA: 'METADATA:' .*;
// Parsing

// A Relay program is a list of global definitions or an expression.
prog: SEMVER (defn* | expr) METADATA? EOF ;

// option: 'set' ident BOOL_LIT ;

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
  | expr '.' NAT                              # projection
  | '[' (expr (',' expr)*)? ']'               # tensor
  | 'if' '(' expr ')' body 'else' body        # ifElse
  // sequencing
  | 'let' var '=' expr ';' expr               # let
  // sugar for let %_ = expr; expr
  | expr ';;' expr                            # let
  | GRAPH_VAR '=' expr ';' expr               # graph
  | ident                                     # identExpr
  | scalar                                    # scalarExpr
  | meta                                      # metaExpr
  | QUOTED_STRING                             # stringExpr
  ;

func: 'fn'        typeParamList? '(' argList ')' ('->' type_)? body ;
defn: 'def' ident typeParamList? '(' argList ')' ('->' type_)? body ;

argList
  : varList              # argNoAttr
  | (var ',')* attrSeq   # argWithAttr
  ;

varList: (var (',' var)*)?;
var: LOCAL_VAR (':' type_)?;

attrSeq: attr (',' attr)*;
attr: CNAME '=' expr ;

typeParamList
  : '[' ']'
  | '[' ident (',' ident)* ']'
  ;

type_
  : '(' ')'                                                      # tupleType
  | '(' type_ ',' ')'                                            # tupleType
  | '(' type_ (',' type_)+ ')'                                   # tupleType
  | typeIdent                                                    # typeIdentType
  | 'Tensor' '[' shapeList ',' type_ ']'                         # tensorType
  | 'fn' typeParamList? '(' (type_ (',' type_)*)? ')' '->' type_ # funcType
  | '_'                                                          # incompleteType
  | NAT                                                          # intType
  ;

shapeList
  : '(' shape (',' shape)+ ')'
  | '(' ')'
  | shape
  ;

meta : 'meta' '[' CNAME ']' '[' NAT ']';

shape
  : meta # metaShape
  | '(' shape ')'                        # parensShape
  | NAT                                  # intShape
  ;

typeIdent : CNAME;
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
  | GLOBAL_VAR
  | LOCAL_VAR
  | GRAPH_VAR
  ;
