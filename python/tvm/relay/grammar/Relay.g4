grammar Relay;

SEMVER: 'v0.0.1' ;

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
FLOAT
  : NAT '.' NAT EXP? // 1.35, 1.35E-9, 0.3, 4.5
  | NAT EXP // 1e10 3e4
  ;

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

func: 'fn'        '(' argList ')' ('->' type_)? body ;
defn: 'def' ident '(' argList ')' ('->' type_)? body ;

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

type_
  : '(' ')'                                         # tupleType
  | '(' type_ ',' ')'                               # tupleType
  | '(' type_ (',' type_)+ ')'                      # tupleType
  | identType                                       # identTypeType
  | 'Tensor' '[' shapeSeq ',' type_ ']'             # tensorType
  // currently unused
  // | identType '[' (type_ (',' type_)*)? ']'         # callType
  | 'fn' '(' (type_ (',' type_)*)? ')' '->' type_   # funcType
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

identType: CNAME ;
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
