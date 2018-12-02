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
LOCAL_VAR: '%' CNAME ;

MUT: 'mut' ;

BOOL_LIT
  : 'True'
  | 'False'
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

// A Relay program is a list of global definitions or an expression.
prog: (defn* | expr) EOF ;

// option: 'set' ident BOOL_LIT ;

expr
  // operators
  : '(' expr ')'                              # parens
  | '-' expr                                  # neg
  | expr op=('*'|'/') expr                    # binOp
  | expr op=('+'|'-') expr                    # binOp
  | expr op=('<'|'>'|'<='|'>=') expr          # binOp
  | expr op=('=='|'!=') expr                  # binOp

  // function definition and application
  | expr '(' (expr (',' expr)*)? ')'          # call
  | func                                      # funcExpr

  // tuples and tensors
  | '(' ')'                                   # tuple
  | '(' expr ',' ')'                          # tuple
  | '(' expr (',' expr)+ ')'                  # tuple
  | '[' (expr (',' expr)*)? ']'               # tensor

  | 'if' '(' expr ')' body 'else' body        # ifElse

  // sequencing
  | 'let' MUT? var '=' expr ';' expr          # seq
  | 'let' MUT? var '=' '{' expr '}' ';' expr  # seq
  // sugar for let %_ = expr; expr
  | expr ';' expr                             # seq

  // mutable update
  // | ident '=' expr                            # writeRef
  // | expr '^'                                  # readRef

  | ident                                     # identExpr
  | scalar                                    # scalarExpr
  // | expr '.' INT                              # project
  // | 'debug'                                   # debug
  ;

func: 'fn'        varList ('->' type_)? body ;
defn: 'def' ident varList ('->' type_)? body ;

varList: '(' (var (',' var)*)? ')' ;
var: ident (':' type_)? ;

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
  | INT                                             # intType
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
  | INT                             # intShape
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
  | LOCAL_VAR
  ;
