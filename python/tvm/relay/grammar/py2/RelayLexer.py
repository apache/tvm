# Generated from /home/sslyu/tvm/python/tvm/relay/grammar/Relay.g4 by ANTLR 4.7.2
# encoding: utf-8
from __future__ import print_function
from antlr4 import *
from io import StringIO
import sys



def serializedATN():
    with StringIO() as buf:
        buf.write(u"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2")
        buf.write(u"*\u010b\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4")
        buf.write(u"\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r")
        buf.write(u"\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22")
        buf.write(u"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4")
        buf.write(u"\30\t\30\4\31\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35")
        buf.write(u"\t\35\4\36\t\36\4\37\t\37\4 \t \4!\t!\4\"\t\"\4#\t#\4")
        buf.write(u"$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t+\4,\t")
        buf.write(u",\3\2\3\2\3\3\3\3\3\4\3\4\3\5\3\5\3\6\3\6\3\7\3\7\3\7")
        buf.write(u"\3\b\3\b\3\b\3\b\3\b\3\t\3\t\3\t\3\t\3\n\3\n\3\13\3\13")
        buf.write(u"\3\f\3\f\3\r\3\r\3\16\3\16\3\16\3\17\3\17\3\17\3\20\3")
        buf.write(u"\20\3\20\3\20\3\21\3\21\3\22\3\22\3\22\3\22\3\22\3\22")
        buf.write(u"\3\22\3\23\3\23\3\24\3\24\3\24\3\24\3\24\3\24\3\24\3")
        buf.write(u"\25\6\25\u0095\n\25\r\25\16\25\u0096\3\25\3\25\3\26\3")
        buf.write(u"\26\3\26\3\26\7\26\u009f\n\26\f\26\16\26\u00a2\13\26")
        buf.write(u"\3\26\3\26\3\26\3\26\3\27\3\27\3\27\3\27\7\27\u00ac\n")
        buf.write(u"\27\f\27\16\27\u00af\13\27\3\27\3\27\3\27\3\27\3\27\3")
        buf.write(u"\30\3\30\3\31\3\31\3\32\3\32\3\33\3\33\3\34\3\34\3\35")
        buf.write(u"\3\35\3\36\3\36\3\36\3\37\3\37\3\37\3 \3 \3 \3!\3!\3")
        buf.write(u"!\3\"\3\"\3\"\3#\3#\3#\3$\3$\3$\3%\3%\3%\3%\3&\3&\3&")
        buf.write(u"\3&\3&\3&\3&\3&\3&\5&\u00e4\n&\3\'\3\'\3\'\3\'\5\'\u00ea")
        buf.write(u"\n\'\3\'\3\'\3\'\5\'\u00ef\n\'\3(\6(\u00f2\n(\r(\16(")
        buf.write(u"\u00f3\3)\3)\5)\u00f8\n)\3)\3)\3*\3*\5*\u00fe\n*\3*\3")
        buf.write(u"*\3*\7*\u0103\n*\f*\16*\u0106\13*\3+\3+\3,\3,\4\u00a0")
        buf.write(u"\u00ad\2-\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25")
        buf.write(u"\f\27\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26")
        buf.write(u"+\27-\30/\31\61\32\63\33\65\34\67\359\36;\37= ?!A\"C")
        buf.write(u"#E$G%I&K\'M(O)Q\2S*U\2W\2\3\2\7\5\2\13\f\17\17\"\"\4")
        buf.write(u"\2GGgg\4\2--//\4\2C\\c|\3\2\62;\2\u0113\2\3\3\2\2\2\2")
        buf.write(u"\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3")
        buf.write(u"\2\2\2\2\17\3\2\2\2\2\21\3\2\2\2\2\23\3\2\2\2\2\25\3")
        buf.write(u"\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2\2\2\2\35\3")
        buf.write(u"\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2")
        buf.write(u"\2\'\3\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2")
        buf.write(u"\2\2\2\61\3\2\2\2\2\63\3\2\2\2\2\65\3\2\2\2\2\67\3\2")
        buf.write(u"\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3\2\2\2\2A\3")
        buf.write(u"\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3\2\2\2\2")
        buf.write(u"K\3\2\2\2\2M\3\2\2\2\2O\3\2\2\2\2S\3\2\2\2\3Y\3\2\2\2")
        buf.write(u"\5[\3\2\2\2\7]\3\2\2\2\t_\3\2\2\2\13a\3\2\2\2\rc\3\2")
        buf.write(u"\2\2\17f\3\2\2\2\21k\3\2\2\2\23o\3\2\2\2\25q\3\2\2\2")
        buf.write(u"\27s\3\2\2\2\31u\3\2\2\2\33w\3\2\2\2\35z\3\2\2\2\37}")
        buf.write(u"\3\2\2\2!\u0081\3\2\2\2#\u0083\3\2\2\2%\u008a\3\2\2\2")
        buf.write(u"\'\u008c\3\2\2\2)\u0094\3\2\2\2+\u009a\3\2\2\2-\u00a7")
        buf.write(u"\3\2\2\2/\u00b5\3\2\2\2\61\u00b7\3\2\2\2\63\u00b9\3\2")
        buf.write(u"\2\2\65\u00bb\3\2\2\2\67\u00bd\3\2\2\29\u00bf\3\2\2\2")
        buf.write(u";\u00c1\3\2\2\2=\u00c4\3\2\2\2?\u00c7\3\2\2\2A\u00ca")
        buf.write(u"\3\2\2\2C\u00cd\3\2\2\2E\u00d0\3\2\2\2G\u00d3\3\2\2\2")
        buf.write(u"I\u00d6\3\2\2\2K\u00e3\3\2\2\2M\u00ee\3\2\2\2O\u00f1")
        buf.write(u"\3\2\2\2Q\u00f5\3\2\2\2S\u00fd\3\2\2\2U\u0107\3\2\2\2")
        buf.write(u"W\u0109\3\2\2\2YZ\7*\2\2Z\4\3\2\2\2[\\\7+\2\2\\\6\3\2")
        buf.write(u"\2\2]^\7.\2\2^\b\3\2\2\2_`\7]\2\2`\n\3\2\2\2ab\7_\2\2")
        buf.write(u"b\f\3\2\2\2cd\7k\2\2de\7h\2\2e\16\3\2\2\2fg\7g\2\2gh")
        buf.write(u"\7n\2\2hi\7u\2\2ij\7g\2\2j\20\3\2\2\2kl\7n\2\2lm\7g\2")
        buf.write(u"\2mn\7v\2\2n\22\3\2\2\2op\7?\2\2p\24\3\2\2\2qr\7=\2\2")
        buf.write(u"r\26\3\2\2\2st\7}\2\2t\30\3\2\2\2uv\7\177\2\2v\32\3\2")
        buf.write(u"\2\2wx\7h\2\2xy\7p\2\2y\34\3\2\2\2z{\7/\2\2{|\7@\2\2")
        buf.write(u"|\36\3\2\2\2}~\7f\2\2~\177\7g\2\2\177\u0080\7h\2\2\u0080")
        buf.write(u" \3\2\2\2\u0081\u0082\7<\2\2\u0082\"\3\2\2\2\u0083\u0084")
        buf.write(u"\7V\2\2\u0084\u0085\7g\2\2\u0085\u0086\7p\2\2\u0086\u0087")
        buf.write(u"\7u\2\2\u0087\u0088\7q\2\2\u0088\u0089\7t\2\2\u0089$")
        buf.write(u"\3\2\2\2\u008a\u008b\7a\2\2\u008b&\3\2\2\2\u008c\u008d")
        buf.write(u"\7x\2\2\u008d\u008e\7\62\2\2\u008e\u008f\7\60\2\2\u008f")
        buf.write(u"\u0090\7\62\2\2\u0090\u0091\7\60\2\2\u0091\u0092\7\64")
        buf.write(u"\2\2\u0092(\3\2\2\2\u0093\u0095\t\2\2\2\u0094\u0093\3")
        buf.write(u"\2\2\2\u0095\u0096\3\2\2\2\u0096\u0094\3\2\2\2\u0096")
        buf.write(u"\u0097\3\2\2\2\u0097\u0098\3\2\2\2\u0098\u0099\b\25\2")
        buf.write(u"\2\u0099*\3\2\2\2\u009a\u009b\7\61\2\2\u009b\u009c\7")
        buf.write(u"\61\2\2\u009c\u00a0\3\2\2\2\u009d\u009f\13\2\2\2\u009e")
        buf.write(u"\u009d\3\2\2\2\u009f\u00a2\3\2\2\2\u00a0\u00a1\3\2\2")
        buf.write(u"\2\u00a0\u009e\3\2\2\2\u00a1\u00a3\3\2\2\2\u00a2\u00a0")
        buf.write(u"\3\2\2\2\u00a3\u00a4\7\f\2\2\u00a4\u00a5\3\2\2\2\u00a5")
        buf.write(u"\u00a6\b\26\2\2\u00a6,\3\2\2\2\u00a7\u00a8\7\61\2\2\u00a8")
        buf.write(u"\u00a9\7,\2\2\u00a9\u00ad\3\2\2\2\u00aa\u00ac\13\2\2")
        buf.write(u"\2\u00ab\u00aa\3\2\2\2\u00ac\u00af\3\2\2\2\u00ad\u00ae")
        buf.write(u"\3\2\2\2\u00ad\u00ab\3\2\2\2\u00ae\u00b0\3\2\2\2\u00af")
        buf.write(u"\u00ad\3\2\2\2\u00b0\u00b1\7,\2\2\u00b1\u00b2\7\61\2")
        buf.write(u"\2\u00b2\u00b3\3\2\2\2\u00b3\u00b4\b\27\2\2\u00b4.\3")
        buf.write(u"\2\2\2\u00b5\u00b6\7,\2\2\u00b6\60\3\2\2\2\u00b7\u00b8")
        buf.write(u"\7\61\2\2\u00b8\62\3\2\2\2\u00b9\u00ba\7-\2\2\u00ba\64")
        buf.write(u"\3\2\2\2\u00bb\u00bc\7/\2\2\u00bc\66\3\2\2\2\u00bd\u00be")
        buf.write(u"\7>\2\2\u00be8\3\2\2\2\u00bf\u00c0\7@\2\2\u00c0:\3\2")
        buf.write(u"\2\2\u00c1\u00c2\7>\2\2\u00c2\u00c3\7?\2\2\u00c3<\3\2")
        buf.write(u"\2\2\u00c4\u00c5\7@\2\2\u00c5\u00c6\7?\2\2\u00c6>\3\2")
        buf.write(u"\2\2\u00c7\u00c8\7?\2\2\u00c8\u00c9\7?\2\2\u00c9@\3\2")
        buf.write(u"\2\2\u00ca\u00cb\7#\2\2\u00cb\u00cc\7?\2\2\u00ccB\3\2")
        buf.write(u"\2\2\u00cd\u00ce\7B\2\2\u00ce\u00cf\5S*\2\u00cfD\3\2")
        buf.write(u"\2\2\u00d0\u00d1\7\'\2\2\u00d1\u00d2\5S*\2\u00d2F\3\2")
        buf.write(u"\2\2\u00d3\u00d4\7\'\2\2\u00d4\u00d5\5O(\2\u00d5H\3\2")
        buf.write(u"\2\2\u00d6\u00d7\7o\2\2\u00d7\u00d8\7w\2\2\u00d8\u00d9")
        buf.write(u"\7v\2\2\u00d9J\3\2\2\2\u00da\u00db\7V\2\2\u00db\u00dc")
        buf.write(u"\7t\2\2\u00dc\u00dd\7w\2\2\u00dd\u00e4\7g\2\2\u00de\u00df")
        buf.write(u"\7H\2\2\u00df\u00e0\7c\2\2\u00e0\u00e1\7n\2\2\u00e1\u00e2")
        buf.write(u"\7u\2\2\u00e2\u00e4\7g\2\2\u00e3\u00da\3\2\2\2\u00e3")
        buf.write(u"\u00de\3\2\2\2\u00e4L\3\2\2\2\u00e5\u00e6\5O(\2\u00e6")
        buf.write(u"\u00e7\7\60\2\2\u00e7\u00e9\5O(\2\u00e8\u00ea\5Q)\2\u00e9")
        buf.write(u"\u00e8\3\2\2\2\u00e9\u00ea\3\2\2\2\u00ea\u00ef\3\2\2")
        buf.write(u"\2\u00eb\u00ec\5O(\2\u00ec\u00ed\5Q)\2\u00ed\u00ef\3")
        buf.write(u"\2\2\2\u00ee\u00e5\3\2\2\2\u00ee\u00eb\3\2\2\2\u00ef")
        buf.write(u"N\3\2\2\2\u00f0\u00f2\5W,\2\u00f1\u00f0\3\2\2\2\u00f2")
        buf.write(u"\u00f3\3\2\2\2\u00f3\u00f1\3\2\2\2\u00f3\u00f4\3\2\2")
        buf.write(u"\2\u00f4P\3\2\2\2\u00f5\u00f7\t\3\2\2\u00f6\u00f8\t\4")
        buf.write(u"\2\2\u00f7\u00f6\3\2\2\2\u00f7\u00f8\3\2\2\2\u00f8\u00f9")
        buf.write(u"\3\2\2\2\u00f9\u00fa\5O(\2\u00faR\3\2\2\2\u00fb\u00fe")
        buf.write(u"\7a\2\2\u00fc\u00fe\5U+\2\u00fd\u00fb\3\2\2\2\u00fd\u00fc")
        buf.write(u"\3\2\2\2\u00fe\u0104\3\2\2\2\u00ff\u0103\7a\2\2\u0100")
        buf.write(u"\u0103\5U+\2\u0101\u0103\5W,\2\u0102\u00ff\3\2\2\2\u0102")
        buf.write(u"\u0100\3\2\2\2\u0102\u0101\3\2\2\2\u0103\u0106\3\2\2")
        buf.write(u"\2\u0104\u0102\3\2\2\2\u0104\u0105\3\2\2\2\u0105T\3\2")
        buf.write(u"\2\2\u0106\u0104\3\2\2\2\u0107\u0108\t\5\2\2\u0108V\3")
        buf.write(u"\2\2\2\u0109\u010a\t\6\2\2\u010aX\3\2\2\2\16\2\u0096")
        buf.write(u"\u00a0\u00ad\u00e3\u00e9\u00ee\u00f3\u00f7\u00fd\u0102")
        buf.write(u"\u0104\3\b\2\2")
        return buf.getvalue()


class RelayLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    T__15 = 16
    T__16 = 17
    T__17 = 18
    SEMVER = 19
    WS = 20
    LINE_COMMENT = 21
    COMMENT = 22
    MUL = 23
    DIV = 24
    ADD = 25
    SUB = 26
    LT = 27
    GT = 28
    LE = 29
    GE = 30
    EQ = 31
    NE = 32
    GLOBAL_VAR = 33
    LOCAL_VAR = 34
    GRAPH_VAR = 35
    MUT = 36
    BOOL_LIT = 37
    FLOAT = 38
    NAT = 39
    CNAME = 40

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ u"DEFAULT_MODE" ]

    literalNames = [ u"<INVALID>",
            u"'('", u"')'", u"','", u"'['", u"']'", u"'if'", u"'else'", 
            u"'let'", u"'='", u"';'", u"'{'", u"'}'", u"'fn'", u"'->'", 
            u"'def'", u"':'", u"'Tensor'", u"'_'", u"'v0.0.2'", u"'*'", 
            u"'/'", u"'+'", u"'-'", u"'<'", u"'>'", u"'<='", u"'>='", u"'=='", 
            u"'!='", u"'mut'" ]

    symbolicNames = [ u"<INVALID>",
            u"SEMVER", u"WS", u"LINE_COMMENT", u"COMMENT", u"MUL", u"DIV", 
            u"ADD", u"SUB", u"LT", u"GT", u"LE", u"GE", u"EQ", u"NE", u"GLOBAL_VAR", 
            u"LOCAL_VAR", u"GRAPH_VAR", u"MUT", u"BOOL_LIT", u"FLOAT", u"NAT", 
            u"CNAME" ]

    ruleNames = [ u"T__0", u"T__1", u"T__2", u"T__3", u"T__4", u"T__5", 
                  u"T__6", u"T__7", u"T__8", u"T__9", u"T__10", u"T__11", 
                  u"T__12", u"T__13", u"T__14", u"T__15", u"T__16", u"T__17", 
                  u"SEMVER", u"WS", u"LINE_COMMENT", u"COMMENT", u"MUL", 
                  u"DIV", u"ADD", u"SUB", u"LT", u"GT", u"LE", u"GE", u"EQ", 
                  u"NE", u"GLOBAL_VAR", u"LOCAL_VAR", u"GRAPH_VAR", u"MUT", 
                  u"BOOL_LIT", u"FLOAT", u"NAT", u"EXP", u"CNAME", u"LETTER", 
                  u"DIGIT" ]

    grammarFileName = u"Relay.g4"

    def __init__(self, input=None, output=sys.stdout):
        super(RelayLexer, self).__init__(input, output=output)
        self.checkVersion("4.7.2")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


