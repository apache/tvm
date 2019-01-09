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
LOCAL_VAR: '%' CNAME;
GRAPH_VAR: '%' INT;

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
  // | expr '.' INT                           # project
  // | 'debug'                                # debug
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
// int8, int16, int32, int64
// uint8, uint16, uint32, uint64
// float16, float32, float64
// bool

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
  | GRAPH_VAR
  ;
