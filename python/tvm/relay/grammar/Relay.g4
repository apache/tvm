grammar Relay;

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
VAR: '%' CNAME ;

MUT: 'mut' ;

BOOL_LIT
  : 'true'
  | 'false'
  ;

// non-negative floats
FLOAT
  : INT '.' INT EXP? // 1.35, 1.35E-9, 0.3, 4.5
  | INT EXP // 1e10 3e4
  ;

// non-negative ints
INT: DIGIT+ ;
fragment EXP: [eE] [+\-]? INT ; // \- since - means "range" inside [...]

CNAME: ('_'|LETTER) ('_'|LETTER|DIGIT)* ;
fragment LETTER: [a-zA-Z] ;
fragment DIGIT: [0-9] ;

// Parsing

// a program is a list of options, a list of global definitions, and an expression
prog: option* defn* expr EOF ;

option: 'set' ident BOOL_LIT ;

expr
  // operators
  : '(' expr ')'                          # parens
  | '-' expr                              # neg
  | expr op=('*'|'/') expr                # binOp
  | expr op=('+'|'-') expr                # binOp
  | expr op=('<'|'>'|'<='|'>=') expr      # binOp
  | expr op=('=='|'!=') expr              # binOp

  // function definition and application
  | expr '(' (expr (',' expr)*)? ')'      # call
  | func                                  # funcExpr

  // tuples and tensors
  | '(' ')'                               # tuple
  | '(' expr ',' ')'                      # tuple
  | '(' expr (',' expr)+ ')'              # tuple
  | '[' (expr (',' expr)*)? ']'           # tensor

  | 'if' '(' expr ')' body 'else' body    # ifElse

  // sequencing
  | 'let' MUT? ident (':' type_)? '=' expr ';' expr  # seq
  // sugar for let _ = expr; expr
  | expr ';' expr                         # seq
  // sugar for let _ = expr; expr
  | '{' expr '}' ';' expr                 # seq

  // mutable update
  // | ident '=' expr                        # writeRef
  // | expr '^'                              # readRef

  | ident                                 # identExpr
  | scalar                                # scalarExpr
  // | expr '.' INT                          # project
  // | 'debug'                               # debug
  ;

func: 'fn'        paramList '->' type_? body ;
defn: 'def' ident paramList '->' type_? body ;

paramList: '(' (param (',' param)*)? ')' ;
param: ident (':' type_)? ;

type_
  // : '(' type_ ')'                           # parensType
  // | type_ op=('*'|'/') type_                # binOpType
  // | type_ op=('+'|'-') type_                # binOpType
  : '(' ')'                                   # tupleType
  | '(' type_ ',' ')'                         # tupleType
  | '(' type_ (',' type_)+ ')'                # tupleType
  | identType                                 # identTypeType
  | identType '[' (type_ (',' type_)*)? ']'   # callType
  | '(' (type_ (',' type_)*)? ')' '->' type_  # funcType
  // Mut, Int, UInt, Float, Bool, Tensor
  | INT                                       # intType
  | '_'                                       # incompleteType
  ;

identType: CNAME ;
// Int8, Int16, Int32, Int64
// UInt8, UInt16, UInt32, UInt64
// Float16, Float32, Float64
// Bool

body: '{' expr '}' ;

scalar
  : FLOAT    # scalarFloat
  | INT      # scalarInt
  | BOOL_LIT # scalarBool
  ;

ident
  : opIdent
  | GLOBAL_VAR
  | VAR
  ;
