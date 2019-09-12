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

/*
 * NOTE: The `USE_ANTLR` option in `config.cmake` must be enabled in order for
 * changes in this file to be reflected by the parser.
 * NOTE: All upper-case rules are *lexer* rules and all camel-case rules are *parser* rules.
 */

grammar Relay;

SEMVER: 'v0.0.4' ;

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

CNAME: ('_'|LETTER) ('_'|LETTER|DIGIT)* ('.' CNAME)* ;

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

// Covers both operator and type idents
generalIdent: CNAME ('.' CNAME)*;
globalVar: '@' CNAME ;
localVar: '%' ('_' | CNAME) ;
graphVar: '%' NAT ;

exprList: (expr (',' expr)*)?;
callList
  : exprList             # callNoAttr
  | (expr ',')* attrSeq  # callWithAttr
  ;

expr
  // operators
  : '(' expr ')'                             # paren
  // function application
  | expr '(' callList ')'                    # call
  | '-' expr                                 # neg
  | expr op=('*'|'/') expr                   # binOp
  | expr op=('+'|'-') expr                   # binOp
  | expr op=('<'|'>'|'<='|'>=') expr         # binOp
  | expr op=('=='|'!=') expr                 # binOp
  // function definition
  | func                                     # funcExpr
  // tuples and tensors
  | '(' ')'                                  # tuple
  | '(' expr ',' ')'                         # tuple
  | '(' expr (',' expr)+ ')'                 # tuple
  | '[' (expr (',' expr)*)? ']'              # tensor
  | 'if' '(' expr ')' body 'else' body       # ifElse
  | matchType expr '{' matchClauseList? '}'  # match
  | expr '.' NAT                             # projection
  // sequencing
  | 'let' var '=' expr ';' expr              # let
  // sugar for let %_ = expr; expr
  | expr ';;' expr                           # let
  | graphVar '=' expr ';' expr               # graph
  | ident                                    # identExpr
  | scalar                                   # scalarExpr
  | meta                                     # metaExpr
  | QUOTED_STRING                            # stringExpr
  ;

func: 'fn' typeParamList? '(' argList ')' ('->' typeExpr)? body ;
defn
  : 'def' globalVar typeParamList? '(' argList ')' ('->' typeExpr)? body  # funcDefn
  | 'extern' 'type' generalIdent typeParamList?                           # externAdtDefn
  | 'type' generalIdent typeParamList? '{' adtConsDefnList? '}'           # adtDefn
  ;

constructorName: CNAME ;

adtConsDefnList: adtConsDefn (',' adtConsDefn)* ','? ;
adtConsDefn: constructorName ('(' typeExpr (',' typeExpr)* ')')? ;
matchClauseList: matchClause (',' matchClause)* ','? ;
matchClause: pattern '=>' ('{' expr '}' | expr) ;
// complete or incomplete match, respectively
matchType : 'match' | 'match?' ;

patternList: '(' pattern (',' pattern)* ')';
pattern
  : '_'                             # wildcardPattern
  | localVar (':' typeExpr)?        # varPattern
  | constructorName patternList?    # constructorPattern
  | patternList                     # tuplePattern
  ;

adtCons: constructorName adtConsParamList? ;
adtConsParamList: '(' adtConsParam (',' adtConsParam)* ')' ;
adtConsParam: localVar | constructorName ;

argList
  : varList             # argNoAttr
  | (var ',')* attrSeq  # argWithAttr
  ;

varList: (var (',' var)*)? ;
var: localVar (':' typeExpr)? ;

attrSeq: attr (',' attr)* ;
attr: CNAME '=' expr ;

typeExpr
  : '(' ')'                                                                # tupleType
  | '(' typeExpr ')'                                                       # typeParen
  | '(' typeExpr ',' ')'                                                   # tupleType
  | '(' typeExpr (',' typeExpr)+ ')'                                       # tupleType
  | generalIdent typeParamList                                             # typeCallType
  | generalIdent                                                           # typeIdentType
  | 'Tensor' '[' shapeList ',' typeExpr ']'                                # tensorType
  | 'fn' typeParamList? '(' (typeExpr (',' typeExpr)*)? ')' '->' typeExpr  # funcType
  | '_'                                                                    # incompleteType
  ;

typeParamList: '[' typeExpr (',' typeExpr)* ']' ;

shapeList
  : '(' ')'
  | '(' shape (',' shape)+ ')'
  | shape
  ;

meta : 'meta' '[' CNAME ']' '[' NAT ']';

shape
  : meta           # metaShape
  | '(' shape ')'  # parensShape
  | NAT            # intShape
  ;

body: '{' expr '}' ;

scalar
  : FLOAT     # scalarFloat
  | NAT       # scalarInt
  | BOOL_LIT  # scalarBool
  ;

ident
  : generalIdent
  | globalVar
  | localVar
  | graphVar
  ;
