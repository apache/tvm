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

grammar Relay;

SEMVER: 'v0.0.3' ;

// Lexing
// comments
WS : [ \t\n\r]+ -> skip ;
LINE_COMMENT : '//' .*? '\n' -> skip ;
COMMENT : '/*' .*? '*/' -> skip ;

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

opIdent: CNAME ;
GLOBAL_VAR: '@' CNAME ;
LOCAL_VAR: '%' CNAME;
GRAPH_VAR: '%' NAT;

MUT: 'mut' ;

BOOL_LIT
  : 'True'
  | 'False'
  ;

// non-negative floats
fragment PREFLOAT : NAT ('.' NAT)? EXP?; // 1.35, 1.35E-9, 0.3, 4.5, 1, 1e10 3e4

FLOAT : PREFLOAT 'f';

// non-negative ints
NAT: DIGIT+ ;
fragment EXP: [eE] [+\-]? NAT ; // \- since - means "range" inside [...]

CNAME: ('_'|LETTER) ('_'|LETTER|DIGIT)* ;
fragment LETTER: [a-zA-Z] ;
fragment DIGIT: [0-9] ;

// Parsing

// A Relay program is a list of global definitions or an expression.
prog: SEMVER (defn* | expr) EOF ;

// option: 'set' ident BOOL_LIT ;

expr
  // operators
  : '(' expr ')'                              # parens
  // function application
  | expr '(' (expr (',' expr)*)? ')'          # call
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

  // sequencing
  | 'let' MUT? var '=' expr ';' expr          # let
  | 'let' MUT? var '=' '{' expr '}' ';' expr  # let
  // sugar for let %_ = expr; expr
  | expr ';' expr                             # let
  | ident '=' expr ';' expr                   # graph

  // mutable update
  // | ident '=' expr                            # writeRef
  // | expr '^'                                  # readRef

  | ident                                     # identExpr
  | scalar                                    # scalarExpr
  // | expr '.' NAT                           # project
  // | 'debug'                                # debug
  ;

func: 'fn'        typeParamSeq? '(' argList ')' ('->' type_)? body ;
defn: 'def' ident typeParamSeq? '(' argList ')' ('->' type_)? body ;

argList
  : varList
  | attrList
  | varList ',' attrList
  ;

varList: (var (',' var)*)? ;
var: ident (':' type_)? ;

attrList: (attr (',' attr)*)? ;
attr: CNAME '=' expr ;

// TODO(@jmp): for improved type annotations
// returnAnno: (ident ':')? type_ ;

// relations: 'where' relation (',' relation)* ;
// relation: ident '(' (type_ (',' type_)*)? ')' ;

typeParamSeq
  : '[' ']'
  | '[' ident (',' ident)* ']'
  ;

type_
  : '(' ')'                                         # tupleType
  | '(' type_ ',' ')'                               # tupleType
  | '(' type_ (',' type_)+ ')'                      # tupleType
  | typeIdent                                       # typeIdentType
  | 'Tensor' '[' shapeSeq ',' type_ ']'             # tensorType
  // currently unused
  // | typeIdent '[' (type_ (',' type_)*)? ']'         # callType
  | 'fn' typeParamSeq? '(' (type_ (',' type_)*)? ')' '->' type_   # funcType
  | '_'                                             # incompleteType
  | NAT                                             # intType
  ;

shapeSeq
  : '(' ')'
  | '(' shape ',' ')'
  | '(' shape (',' shape)+ ')'
  ;

shape
  : '(' shape ')'                   # parensShape
  // | type_ op=('*'|'/') type_        # binOpType
  // | type_ op=('+'|'-') type_        # binOpType
  | NAT                             # intShape
  ;

typeIdent : CNAME ;
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
