# Generated from /home/marisa/Work/tvm/python/tvm/relay/grammar/Relay.g4 by ANTLR 4.7.1
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2*")
        buf.write("\u010d\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7")
        buf.write("\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r")
        buf.write("\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23")
        buf.write("\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30")
        buf.write("\4\31\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36")
        buf.write("\t\36\4\37\t\37\4 \t \4!\t!\4\"\t\"\4#\t#\4$\t$\4%\t%")
        buf.write("\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t+\4,\t,\4-\t-\3\2")
        buf.write("\3\2\3\3\3\3\3\4\3\4\3\5\3\5\3\6\3\6\3\7\3\7\3\7\3\b\3")
        buf.write("\b\3\b\3\b\3\b\3\t\3\t\3\t\3\t\3\n\3\n\3\13\3\13\3\f\3")
        buf.write("\f\3\r\3\r\3\16\3\16\3\16\3\17\3\17\3\17\3\20\3\20\3\20")
        buf.write("\3\20\3\21\3\21\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\23")
        buf.write("\3\23\3\24\3\24\3\24\3\24\3\24\3\24\3\24\3\25\6\25\u0097")
        buf.write("\n\25\r\25\16\25\u0098\3\25\3\25\3\26\3\26\3\26\3\26\7")
        buf.write("\26\u00a1\n\26\f\26\16\26\u00a4\13\26\3\26\3\26\3\26\3")
        buf.write("\26\3\27\3\27\3\27\3\27\7\27\u00ae\n\27\f\27\16\27\u00b1")
        buf.write("\13\27\3\27\3\27\3\27\3\27\3\27\3\30\3\30\3\31\3\31\3")
        buf.write("\32\3\32\3\33\3\33\3\34\3\34\3\35\3\35\3\36\3\36\3\36")
        buf.write("\3\37\3\37\3\37\3 \3 \3 \3!\3!\3!\3\"\3\"\3\"\3#\3#\3")
        buf.write("#\3$\3$\3$\3%\3%\3%\3%\3&\3&\3&\3&\3&\3&\3&\3&\3&\5&\u00e6")
        buf.write("\n&\3\'\3\'\3\'\5\'\u00eb\n\'\3\'\5\'\u00ee\n\'\3(\3(")
        buf.write("\3(\3)\6)\u00f4\n)\r)\16)\u00f5\3*\3*\5*\u00fa\n*\3*\3")
        buf.write("*\3+\3+\5+\u0100\n+\3+\3+\3+\7+\u0105\n+\f+\16+\u0108")
        buf.write("\13+\3,\3,\3-\3-\4\u00a2\u00af\2.\3\3\5\4\7\5\t\6\13\7")
        buf.write("\r\b\17\t\21\n\23\13\25\f\27\r\31\16\33\17\35\20\37\21")
        buf.write("!\22#\23%\24\'\25)\26+\27-\30/\31\61\32\63\33\65\34\67")
        buf.write("\359\36;\37= ?!A\"C#E$G%I&K\'M\2O(Q)S\2U*W\2Y\2\3\2\7")
        buf.write("\5\2\13\f\17\17\"\"\4\2GGgg\4\2--//\4\2C\\c|\3\2\62;\2")
        buf.write("\u0114\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2")
        buf.write("\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3\2\2\2\2")
        buf.write("\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33")
        buf.write("\3\2\2\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2")
        buf.write("\2\2%\3\2\2\2\2\'\3\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2")
        buf.write("\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3\2\2\2\2\65\3\2\2\2")
        buf.write("\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3\2")
        buf.write("\2\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3")
        buf.write("\2\2\2\2K\3\2\2\2\2O\3\2\2\2\2Q\3\2\2\2\2U\3\2\2\2\3[")
        buf.write("\3\2\2\2\5]\3\2\2\2\7_\3\2\2\2\ta\3\2\2\2\13c\3\2\2\2")
        buf.write("\re\3\2\2\2\17h\3\2\2\2\21m\3\2\2\2\23q\3\2\2\2\25s\3")
        buf.write("\2\2\2\27u\3\2\2\2\31w\3\2\2\2\33y\3\2\2\2\35|\3\2\2\2")
        buf.write("\37\177\3\2\2\2!\u0083\3\2\2\2#\u0085\3\2\2\2%\u008c\3")
        buf.write("\2\2\2\'\u008e\3\2\2\2)\u0096\3\2\2\2+\u009c\3\2\2\2-")
        buf.write("\u00a9\3\2\2\2/\u00b7\3\2\2\2\61\u00b9\3\2\2\2\63\u00bb")
        buf.write("\3\2\2\2\65\u00bd\3\2\2\2\67\u00bf\3\2\2\29\u00c1\3\2")
        buf.write("\2\2;\u00c3\3\2\2\2=\u00c6\3\2\2\2?\u00c9\3\2\2\2A\u00cc")
        buf.write("\3\2\2\2C\u00cf\3\2\2\2E\u00d2\3\2\2\2G\u00d5\3\2\2\2")
        buf.write("I\u00d8\3\2\2\2K\u00e5\3\2\2\2M\u00e7\3\2\2\2O\u00ef\3")
        buf.write("\2\2\2Q\u00f3\3\2\2\2S\u00f7\3\2\2\2U\u00ff\3\2\2\2W\u0109")
        buf.write("\3\2\2\2Y\u010b\3\2\2\2[\\\7*\2\2\\\4\3\2\2\2]^\7+\2\2")
        buf.write("^\6\3\2\2\2_`\7.\2\2`\b\3\2\2\2ab\7]\2\2b\n\3\2\2\2cd")
        buf.write("\7_\2\2d\f\3\2\2\2ef\7k\2\2fg\7h\2\2g\16\3\2\2\2hi\7g")
        buf.write("\2\2ij\7n\2\2jk\7u\2\2kl\7g\2\2l\20\3\2\2\2mn\7n\2\2n")
        buf.write("o\7g\2\2op\7v\2\2p\22\3\2\2\2qr\7?\2\2r\24\3\2\2\2st\7")
        buf.write("=\2\2t\26\3\2\2\2uv\7}\2\2v\30\3\2\2\2wx\7\177\2\2x\32")
        buf.write("\3\2\2\2yz\7h\2\2z{\7p\2\2{\34\3\2\2\2|}\7/\2\2}~\7@\2")
        buf.write("\2~\36\3\2\2\2\177\u0080\7f\2\2\u0080\u0081\7g\2\2\u0081")
        buf.write("\u0082\7h\2\2\u0082 \3\2\2\2\u0083\u0084\7<\2\2\u0084")
        buf.write("\"\3\2\2\2\u0085\u0086\7V\2\2\u0086\u0087\7g\2\2\u0087")
        buf.write("\u0088\7p\2\2\u0088\u0089\7u\2\2\u0089\u008a\7q\2\2\u008a")
        buf.write("\u008b\7t\2\2\u008b$\3\2\2\2\u008c\u008d\7a\2\2\u008d")
        buf.write("&\3\2\2\2\u008e\u008f\7x\2\2\u008f\u0090\7\62\2\2\u0090")
        buf.write("\u0091\7\60\2\2\u0091\u0092\7\62\2\2\u0092\u0093\7\60")
        buf.write("\2\2\u0093\u0094\7\65\2\2\u0094(\3\2\2\2\u0095\u0097\t")
        buf.write("\2\2\2\u0096\u0095\3\2\2\2\u0097\u0098\3\2\2\2\u0098\u0096")
        buf.write("\3\2\2\2\u0098\u0099\3\2\2\2\u0099\u009a\3\2\2\2\u009a")
        buf.write("\u009b\b\25\2\2\u009b*\3\2\2\2\u009c\u009d\7\61\2\2\u009d")
        buf.write("\u009e\7\61\2\2\u009e\u00a2\3\2\2\2\u009f\u00a1\13\2\2")
        buf.write("\2\u00a0\u009f\3\2\2\2\u00a1\u00a4\3\2\2\2\u00a2\u00a3")
        buf.write("\3\2\2\2\u00a2\u00a0\3\2\2\2\u00a3\u00a5\3\2\2\2\u00a4")
        buf.write("\u00a2\3\2\2\2\u00a5\u00a6\7\f\2\2\u00a6\u00a7\3\2\2\2")
        buf.write("\u00a7\u00a8\b\26\2\2\u00a8,\3\2\2\2\u00a9\u00aa\7\61")
        buf.write("\2\2\u00aa\u00ab\7,\2\2\u00ab\u00af\3\2\2\2\u00ac\u00ae")
        buf.write("\13\2\2\2\u00ad\u00ac\3\2\2\2\u00ae\u00b1\3\2\2\2\u00af")
        buf.write("\u00b0\3\2\2\2\u00af\u00ad\3\2\2\2\u00b0\u00b2\3\2\2\2")
        buf.write("\u00b1\u00af\3\2\2\2\u00b2\u00b3\7,\2\2\u00b3\u00b4\7")
        buf.write("\61\2\2\u00b4\u00b5\3\2\2\2\u00b5\u00b6\b\27\2\2\u00b6")
        buf.write(".\3\2\2\2\u00b7\u00b8\7,\2\2\u00b8\60\3\2\2\2\u00b9\u00ba")
        buf.write("\7\61\2\2\u00ba\62\3\2\2\2\u00bb\u00bc\7-\2\2\u00bc\64")
        buf.write("\3\2\2\2\u00bd\u00be\7/\2\2\u00be\66\3\2\2\2\u00bf\u00c0")
        buf.write("\7>\2\2\u00c08\3\2\2\2\u00c1\u00c2\7@\2\2\u00c2:\3\2\2")
        buf.write("\2\u00c3\u00c4\7>\2\2\u00c4\u00c5\7?\2\2\u00c5<\3\2\2")
        buf.write("\2\u00c6\u00c7\7@\2\2\u00c7\u00c8\7?\2\2\u00c8>\3\2\2")
        buf.write("\2\u00c9\u00ca\7?\2\2\u00ca\u00cb\7?\2\2\u00cb@\3\2\2")
        buf.write("\2\u00cc\u00cd\7#\2\2\u00cd\u00ce\7?\2\2\u00ceB\3\2\2")
        buf.write("\2\u00cf\u00d0\7B\2\2\u00d0\u00d1\5U+\2\u00d1D\3\2\2\2")
        buf.write("\u00d2\u00d3\7\'\2\2\u00d3\u00d4\5U+\2\u00d4F\3\2\2\2")
        buf.write("\u00d5\u00d6\7\'\2\2\u00d6\u00d7\5Q)\2\u00d7H\3\2\2\2")
        buf.write("\u00d8\u00d9\7o\2\2\u00d9\u00da\7w\2\2\u00da\u00db\7v")
        buf.write("\2\2\u00dbJ\3\2\2\2\u00dc\u00dd\7V\2\2\u00dd\u00de\7t")
        buf.write("\2\2\u00de\u00df\7w\2\2\u00df\u00e6\7g\2\2\u00e0\u00e1")
        buf.write("\7H\2\2\u00e1\u00e2\7c\2\2\u00e2\u00e3\7n\2\2\u00e3\u00e4")
        buf.write("\7u\2\2\u00e4\u00e6\7g\2\2\u00e5\u00dc\3\2\2\2\u00e5\u00e0")
        buf.write("\3\2\2\2\u00e6L\3\2\2\2\u00e7\u00ea\5Q)\2\u00e8\u00e9")
        buf.write("\7\60\2\2\u00e9\u00eb\5Q)\2\u00ea\u00e8\3\2\2\2\u00ea")
        buf.write("\u00eb\3\2\2\2\u00eb\u00ed\3\2\2\2\u00ec\u00ee\5S*\2\u00ed")
        buf.write("\u00ec\3\2\2\2\u00ed\u00ee\3\2\2\2\u00eeN\3\2\2\2\u00ef")
        buf.write("\u00f0\5M\'\2\u00f0\u00f1\7h\2\2\u00f1P\3\2\2\2\u00f2")
        buf.write("\u00f4\5Y-\2\u00f3\u00f2\3\2\2\2\u00f4\u00f5\3\2\2\2\u00f5")
        buf.write("\u00f3\3\2\2\2\u00f5\u00f6\3\2\2\2\u00f6R\3\2\2\2\u00f7")
        buf.write("\u00f9\t\3\2\2\u00f8\u00fa\t\4\2\2\u00f9\u00f8\3\2\2\2")
        buf.write("\u00f9\u00fa\3\2\2\2\u00fa\u00fb\3\2\2\2\u00fb\u00fc\5")
        buf.write("Q)\2\u00fcT\3\2\2\2\u00fd\u0100\7a\2\2\u00fe\u0100\5W")
        buf.write(",\2\u00ff\u00fd\3\2\2\2\u00ff\u00fe\3\2\2\2\u0100\u0106")
        buf.write("\3\2\2\2\u0101\u0105\7a\2\2\u0102\u0105\5W,\2\u0103\u0105")
        buf.write("\5Y-\2\u0104\u0101\3\2\2\2\u0104\u0102\3\2\2\2\u0104\u0103")
        buf.write("\3\2\2\2\u0105\u0108\3\2\2\2\u0106\u0104\3\2\2\2\u0106")
        buf.write("\u0107\3\2\2\2\u0107V\3\2\2\2\u0108\u0106\3\2\2\2\u0109")
        buf.write("\u010a\t\5\2\2\u010aX\3\2\2\2\u010b\u010c\t\6\2\2\u010c")
        buf.write("Z\3\2\2\2\16\2\u0098\u00a2\u00af\u00e5\u00ea\u00ed\u00f5")
        buf.write("\u00f9\u00ff\u0104\u0106\3\b\2\2")
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

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'('", "')'", "','", "'['", "']'", "'if'", "'else'", "'let'", 
            "'='", "';'", "'{'", "'}'", "'fn'", "'->'", "'def'", "':'", 
            "'Tensor'", "'_'", "'v0.0.3'", "'*'", "'/'", "'+'", "'-'", "'<'", 
            "'>'", "'<='", "'>='", "'=='", "'!='", "'mut'" ]

    symbolicNames = [ "<INVALID>",
            "SEMVER", "WS", "LINE_COMMENT", "COMMENT", "MUL", "DIV", "ADD", 
            "SUB", "LT", "GT", "LE", "GE", "EQ", "NE", "GLOBAL_VAR", "LOCAL_VAR", 
            "GRAPH_VAR", "MUT", "BOOL_LIT", "FLOAT", "NAT", "CNAME" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6", 
                  "T__7", "T__8", "T__9", "T__10", "T__11", "T__12", "T__13", 
                  "T__14", "T__15", "T__16", "T__17", "SEMVER", "WS", "LINE_COMMENT", 
                  "COMMENT", "MUL", "DIV", "ADD", "SUB", "LT", "GT", "LE", 
                  "GE", "EQ", "NE", "GLOBAL_VAR", "LOCAL_VAR", "GRAPH_VAR", 
                  "MUT", "BOOL_LIT", "PREFLOAT", "FLOAT", "NAT", "EXP", 
                  "CNAME", "LETTER", "DIGIT" ]

    grammarFileName = "Relay.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


