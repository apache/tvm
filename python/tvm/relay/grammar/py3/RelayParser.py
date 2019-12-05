# Generated from /Users/doobs/Code/repo/sampl/tvm/python/tvm/relay/grammar/Relay.g4 by ANTLR 4.7.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\62")
        buf.write("\u0200\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7")
        buf.write("\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r\4\16")
        buf.write("\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23\t\23")
        buf.write("\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31")
        buf.write("\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36")
        buf.write("\4\37\t\37\4 \t \4!\t!\4\"\t\"\4#\t#\3\2\3\2\7\2I\n\2")
        buf.write("\f\2\16\2L\13\2\3\2\5\2O\n\2\3\2\5\2R\n\2\3\2\3\2\3\3")
        buf.write("\3\3\3\3\7\3Y\n\3\f\3\16\3\\\13\3\3\4\3\4\3\4\3\5\3\5")
        buf.write("\3\5\3\6\3\6\3\6\3\7\3\7\3\7\7\7j\n\7\f\7\16\7m\13\7\5")
        buf.write("\7o\n\7\3\b\3\b\3\b\3\b\7\bu\n\b\f\b\16\bx\13\b\3\b\5")
        buf.write("\b{\n\b\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3")
        buf.write("\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\6\t\u0090\n\t\r\t\16\t")
        buf.write("\u0091\3\t\3\t\3\t\3\t\3\t\3\t\7\t\u009a\n\t\f\t\16\t")
        buf.write("\u009d\13\t\5\t\u009f\n\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t")
        buf.write("\3\t\3\t\3\t\3\t\3\t\3\t\5\t\u00ae\n\t\3\t\3\t\3\t\3\t")
        buf.write("\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3")
        buf.write("\t\3\t\5\t\u00c3\n\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3")
        buf.write("\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t")
        buf.write("\3\t\7\t\u00dc\n\t\f\t\16\t\u00df\13\t\3\n\3\n\5\n\u00e3")
        buf.write("\n\n\3\n\3\n\3\n\3\n\3\n\5\n\u00ea\n\n\3\n\3\n\3\13\3")
        buf.write("\13\3\13\5\13\u00f1\n\13\3\13\3\13\3\13\3\13\3\13\5\13")
        buf.write("\u00f8\n\13\3\13\3\13\3\13\3\13\3\13\3\13\5\13\u0100\n")
        buf.write("\13\3\13\3\13\3\13\5\13\u0105\n\13\3\13\3\13\5\13\u0109")
        buf.write("\n\13\3\13\3\13\5\13\u010d\n\13\3\f\3\f\3\r\3\r\3\r\7")
        buf.write("\r\u0114\n\r\f\r\16\r\u0117\13\r\3\r\5\r\u011a\n\r\3\16")
        buf.write("\3\16\3\16\3\16\3\16\7\16\u0121\n\16\f\16\16\16\u0124")
        buf.write("\13\16\3\16\3\16\5\16\u0128\n\16\3\17\3\17\3\17\7\17\u012d")
        buf.write("\n\17\f\17\16\17\u0130\13\17\3\17\5\17\u0133\n\17\3\20")
        buf.write("\3\20\3\20\3\20\3\20\3\20\3\20\5\20\u013c\n\20\3\21\3")
        buf.write("\21\3\22\3\22\3\22\3\22\7\22\u0144\n\22\f\22\16\22\u0147")
        buf.write("\13\22\3\22\3\22\3\23\3\23\3\23\3\23\5\23\u014f\n\23\3")
        buf.write("\23\3\23\5\23\u0153\n\23\3\23\5\23\u0156\n\23\3\24\3\24")
        buf.write("\5\24\u015a\n\24\3\25\3\25\3\25\3\25\7\25\u0160\n\25\f")
        buf.write("\25\16\25\u0163\13\25\3\25\3\25\3\26\3\26\5\26\u0169\n")
        buf.write("\26\3\27\3\27\3\27\3\27\7\27\u016f\n\27\f\27\16\27\u0172")
        buf.write("\13\27\3\27\5\27\u0175\n\27\3\30\3\30\3\30\7\30\u017a")
        buf.write("\n\30\f\30\16\30\u017d\13\30\5\30\u017f\n\30\3\31\3\31")
        buf.write("\3\31\5\31\u0184\n\31\3\32\3\32\3\32\7\32\u0189\n\32\f")
        buf.write("\32\16\32\u018c\13\32\3\33\3\33\3\33\3\33\3\34\3\34\3")
        buf.write("\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34")
        buf.write("\3\34\3\34\6\34\u01a1\n\34\r\34\16\34\u01a2\3\34\3\34")
        buf.write("\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34")
        buf.write("\3\34\3\34\5\34\u01b4\n\34\3\34\3\34\3\34\3\34\7\34\u01ba")
        buf.write("\n\34\f\34\16\34\u01bd\13\34\5\34\u01bf\n\34\3\34\3\34")
        buf.write("\3\34\3\34\5\34\u01c5\n\34\3\35\3\35\3\35\3\35\7\35\u01cb")
        buf.write("\n\35\f\35\16\35\u01ce\13\35\3\35\3\35\3\36\3\36\3\36")
        buf.write("\3\36\3\36\3\36\6\36\u01d8\n\36\r\36\16\36\u01d9\3\36")
        buf.write("\3\36\3\36\5\36\u01df\n\36\3\37\3\37\3\37\3\37\3\37\3")
        buf.write("\37\3\37\3\37\3 \3 \3 \3 \3 \3 \5 \u01ef\n \3!\3!\3!\3")
        buf.write("!\3\"\3\"\3\"\5\"\u01f8\n\"\3#\3#\3#\3#\5#\u01fe\n#\3")
        buf.write("#\2\3\20$\2\4\6\b\n\f\16\20\22\24\26\30\32\34\36 \"$&")
        buf.write("(*,.\60\62\64\668:<>@BD\2\b\4\2\6\6//\3\2$%\3\2&\'\3\2")
        buf.write("(+\3\2,-\3\2\32\33\2\u0234\2F\3\2\2\2\4U\3\2\2\2\6]\3")
        buf.write("\2\2\2\b`\3\2\2\2\nc\3\2\2\2\fn\3\2\2\2\16z\3\2\2\2\20")
        buf.write("\u00c2\3\2\2\2\22\u00e0\3\2\2\2\24\u010c\3\2\2\2\26\u010e")
        buf.write("\3\2\2\2\30\u0110\3\2\2\2\32\u011b\3\2\2\2\34\u0129\3")
        buf.write("\2\2\2\36\u0134\3\2\2\2 \u013d\3\2\2\2\"\u013f\3\2\2\2")
        buf.write("$\u0155\3\2\2\2&\u0157\3\2\2\2(\u015b\3\2\2\2*\u0168\3")
        buf.write("\2\2\2,\u0174\3\2\2\2.\u017e\3\2\2\2\60\u0180\3\2\2\2")
        buf.write("\62\u0185\3\2\2\2\64\u018d\3\2\2\2\66\u01c4\3\2\2\28\u01c6")
        buf.write("\3\2\2\2:\u01de\3\2\2\2<\u01e0\3\2\2\2>\u01ee\3\2\2\2")
        buf.write("@\u01f0\3\2\2\2B\u01f7\3\2\2\2D\u01fd\3\2\2\2FN\7\37\2")
        buf.write("\2GI\5\24\13\2HG\3\2\2\2IL\3\2\2\2JH\3\2\2\2JK\3\2\2\2")
        buf.write("KO\3\2\2\2LJ\3\2\2\2MO\5\20\t\2NJ\3\2\2\2NM\3\2\2\2OQ")
        buf.write("\3\2\2\2PR\7\62\2\2QP\3\2\2\2QR\3\2\2\2RS\3\2\2\2ST\7")
        buf.write("\2\2\3T\3\3\2\2\2UZ\7/\2\2VW\7\3\2\2WY\7/\2\2XV\3\2\2")
        buf.write("\2Y\\\3\2\2\2ZX\3\2\2\2Z[\3\2\2\2[\5\3\2\2\2\\Z\3\2\2")
        buf.write("\2]^\7\4\2\2^_\7/\2\2_\7\3\2\2\2`a\7\5\2\2ab\t\2\2\2b")
        buf.write("\t\3\2\2\2cd\7\5\2\2de\7\61\2\2e\13\3\2\2\2fk\5\20\t\2")
        buf.write("gh\7\7\2\2hj\5\20\t\2ig\3\2\2\2jm\3\2\2\2ki\3\2\2\2kl")
        buf.write("\3\2\2\2lo\3\2\2\2mk\3\2\2\2nf\3\2\2\2no\3\2\2\2o\r\3")
        buf.write("\2\2\2p{\5\f\7\2qr\5\20\t\2rs\7\7\2\2su\3\2\2\2tq\3\2")
        buf.write("\2\2ux\3\2\2\2vt\3\2\2\2vw\3\2\2\2wy\3\2\2\2xv\3\2\2\2")
        buf.write("y{\5\62\32\2zp\3\2\2\2zv\3\2\2\2{\17\3\2\2\2|}\b\t\1\2")
        buf.write("}~\7\b\2\2~\177\5\20\t\2\177\u0080\7\t\2\2\u0080\u00c3")
        buf.write("\3\2\2\2\u0081\u0082\7\'\2\2\u0082\u00c3\5\20\t\26\u0083")
        buf.write("\u00c3\5\22\n\2\u0084\u0085\7\b\2\2\u0085\u00c3\7\t\2")
        buf.write("\2\u0086\u0087\7\b\2\2\u0087\u0088\5\20\t\2\u0088\u0089")
        buf.write("\7\7\2\2\u0089\u008a\7\t\2\2\u008a\u00c3\3\2\2\2\u008b")
        buf.write("\u008c\7\b\2\2\u008c\u008f\5\20\t\2\u008d\u008e\7\7\2")
        buf.write("\2\u008e\u0090\5\20\t\2\u008f\u008d\3\2\2\2\u0090\u0091")
        buf.write("\3\2\2\2\u0091\u008f\3\2\2\2\u0091\u0092\3\2\2\2\u0092")
        buf.write("\u0093\3\2\2\2\u0093\u0094\7\t\2\2\u0094\u00c3\3\2\2\2")
        buf.write("\u0095\u009e\7\n\2\2\u0096\u009b\5\20\t\2\u0097\u0098")
        buf.write("\7\7\2\2\u0098\u009a\5\20\t\2\u0099\u0097\3\2\2\2\u009a")
        buf.write("\u009d\3\2\2\2\u009b\u0099\3\2\2\2\u009b\u009c\3\2\2\2")
        buf.write("\u009c\u009f\3\2\2\2\u009d\u009b\3\2\2\2\u009e\u0096\3")
        buf.write("\2\2\2\u009e\u009f\3\2\2\2\u009f\u00a0\3\2\2\2\u00a0\u00c3")
        buf.write("\7\13\2\2\u00a1\u00a2\7\f\2\2\u00a2\u00a3\7\b\2\2\u00a3")
        buf.write("\u00a4\5\20\t\2\u00a4\u00a5\7\t\2\2\u00a5\u00a6\5@!\2")
        buf.write("\u00a6\u00a7\7\r\2\2\u00a7\u00a8\5@!\2\u00a8\u00c3\3\2")
        buf.write("\2\2\u00a9\u00aa\5 \21\2\u00aa\u00ab\5\20\t\2\u00ab\u00ad")
        buf.write("\7\16\2\2\u00ac\u00ae\5\34\17\2\u00ad\u00ac\3\2\2\2\u00ad")
        buf.write("\u00ae\3\2\2\2\u00ae\u00af\3\2\2\2\u00af\u00b0\7\17\2")
        buf.write("\2\u00b0\u00c3\3\2\2\2\u00b1\u00b2\7\20\2\2\u00b2\u00b3")
        buf.write("\5\60\31\2\u00b3\u00b4\7\21\2\2\u00b4\u00b5\5\20\t\2\u00b5")
        buf.write("\u00b6\7\22\2\2\u00b6\u00b7\5\20\t\t\u00b7\u00c3\3\2\2")
        buf.write("\2\u00b8\u00b9\5\n\6\2\u00b9\u00ba\7\21\2\2\u00ba\u00bb")
        buf.write("\5\20\t\2\u00bb\u00bc\7\22\2\2\u00bc\u00bd\5\20\t\7\u00bd")
        buf.write("\u00c3\3\2\2\2\u00be\u00c3\5D#\2\u00bf\u00c3\5B\"\2\u00c0")
        buf.write("\u00c3\5<\37\2\u00c1\u00c3\7#\2\2\u00c2|\3\2\2\2\u00c2")
        buf.write("\u0081\3\2\2\2\u00c2\u0083\3\2\2\2\u00c2\u0084\3\2\2\2")
        buf.write("\u00c2\u0086\3\2\2\2\u00c2\u008b\3\2\2\2\u00c2\u0095\3")
        buf.write("\2\2\2\u00c2\u00a1\3\2\2\2\u00c2\u00a9\3\2\2\2\u00c2\u00b1")
        buf.write("\3\2\2\2\u00c2\u00b8\3\2\2\2\u00c2\u00be\3\2\2\2\u00c2")
        buf.write("\u00bf\3\2\2\2\u00c2\u00c0\3\2\2\2\u00c2\u00c1\3\2\2\2")
        buf.write("\u00c3\u00dd\3\2\2\2\u00c4\u00c5\f\25\2\2\u00c5\u00c6")
        buf.write("\t\3\2\2\u00c6\u00dc\5\20\t\26\u00c7\u00c8\f\24\2\2\u00c8")
        buf.write("\u00c9\t\4\2\2\u00c9\u00dc\5\20\t\25\u00ca\u00cb\f\23")
        buf.write("\2\2\u00cb\u00cc\t\5\2\2\u00cc\u00dc\5\20\t\24\u00cd\u00ce")
        buf.write("\f\22\2\2\u00ce\u00cf\t\6\2\2\u00cf\u00dc\5\20\t\23\u00d0")
        buf.write("\u00d1\f\b\2\2\u00d1\u00d2\7\23\2\2\u00d2\u00dc\5\20\t")
        buf.write("\t\u00d3\u00d4\f\27\2\2\u00d4\u00d5\7\b\2\2\u00d5\u00d6")
        buf.write("\5\16\b\2\u00d6\u00d7\7\t\2\2\u00d7\u00dc\3\2\2\2\u00d8")
        buf.write("\u00d9\f\n\2\2\u00d9\u00da\7\3\2\2\u00da\u00dc\7\61\2")
        buf.write("\2\u00db\u00c4\3\2\2\2\u00db\u00c7\3\2\2\2\u00db\u00ca")
        buf.write("\3\2\2\2\u00db\u00cd\3\2\2\2\u00db\u00d0\3\2\2\2\u00db")
        buf.write("\u00d3\3\2\2\2\u00db\u00d8\3\2\2\2\u00dc\u00df\3\2\2\2")
        buf.write("\u00dd\u00db\3\2\2\2\u00dd\u00de\3\2\2\2\u00de\21\3\2")
        buf.write("\2\2\u00df\u00dd\3\2\2\2\u00e0\u00e2\7\24\2\2\u00e1\u00e3")
        buf.write("\58\35\2\u00e2\u00e1\3\2\2\2\u00e2\u00e3\3\2\2\2\u00e3")
        buf.write("\u00e4\3\2\2\2\u00e4\u00e5\7\b\2\2\u00e5\u00e6\5,\27\2")
        buf.write("\u00e6\u00e9\7\t\2\2\u00e7\u00e8\7\25\2\2\u00e8\u00ea")
        buf.write("\5\66\34\2\u00e9\u00e7\3\2\2\2\u00e9\u00ea\3\2\2\2\u00ea")
        buf.write("\u00eb\3\2\2\2\u00eb\u00ec\5@!\2\u00ec\23\3\2\2\2\u00ed")
        buf.write("\u00ee\7\26\2\2\u00ee\u00f0\5\6\4\2\u00ef\u00f1\58\35")
        buf.write("\2\u00f0\u00ef\3\2\2\2\u00f0\u00f1\3\2\2\2\u00f1\u00f2")
        buf.write("\3\2\2\2\u00f2\u00f3\7\b\2\2\u00f3\u00f4\5,\27\2\u00f4")
        buf.write("\u00f7\7\t\2\2\u00f5\u00f6\7\25\2\2\u00f6\u00f8\5\66\34")
        buf.write("\2\u00f7\u00f5\3\2\2\2\u00f7\u00f8\3\2\2\2\u00f8\u00f9")
        buf.write("\3\2\2\2\u00f9\u00fa\5@!\2\u00fa\u010d\3\2\2\2\u00fb\u00fc")
        buf.write("\7\27\2\2\u00fc\u00fd\7\30\2\2\u00fd\u00ff\5\4\3\2\u00fe")
        buf.write("\u0100\58\35\2\u00ff\u00fe\3\2\2\2\u00ff\u0100\3\2\2\2")
        buf.write("\u0100\u010d\3\2\2\2\u0101\u0102\7\30\2\2\u0102\u0104")
        buf.write("\5\4\3\2\u0103\u0105\58\35\2\u0104\u0103\3\2\2\2\u0104")
        buf.write("\u0105\3\2\2\2\u0105\u0106\3\2\2\2\u0106\u0108\7\16\2")
        buf.write("\2\u0107\u0109\5\30\r\2\u0108\u0107\3\2\2\2\u0108\u0109")
        buf.write("\3\2\2\2\u0109\u010a\3\2\2\2\u010a\u010b\7\17\2\2\u010b")
        buf.write("\u010d\3\2\2\2\u010c\u00ed\3\2\2\2\u010c\u00fb\3\2\2\2")
        buf.write("\u010c\u0101\3\2\2\2\u010d\25\3\2\2\2\u010e\u010f\7/\2")
        buf.write("\2\u010f\27\3\2\2\2\u0110\u0115\5\32\16\2\u0111\u0112")
        buf.write("\7\7\2\2\u0112\u0114\5\32\16\2\u0113\u0111\3\2\2\2\u0114")
        buf.write("\u0117\3\2\2\2\u0115\u0113\3\2\2\2\u0115\u0116\3\2\2\2")
        buf.write("\u0116\u0119\3\2\2\2\u0117\u0115\3\2\2\2\u0118\u011a\7")
        buf.write("\7\2\2\u0119\u0118\3\2\2\2\u0119\u011a\3\2\2\2\u011a\31")
        buf.write("\3\2\2\2\u011b\u0127\5\26\f\2\u011c\u011d\7\b\2\2\u011d")
        buf.write("\u0122\5\66\34\2\u011e\u011f\7\7\2\2\u011f\u0121\5\66")
        buf.write("\34\2\u0120\u011e\3\2\2\2\u0121\u0124\3\2\2\2\u0122\u0120")
        buf.write("\3\2\2\2\u0122\u0123\3\2\2\2\u0123\u0125\3\2\2\2\u0124")
        buf.write("\u0122\3\2\2\2\u0125\u0126\7\t\2\2\u0126\u0128\3\2\2\2")
        buf.write("\u0127\u011c\3\2\2\2\u0127\u0128\3\2\2\2\u0128\33\3\2")
        buf.write("\2\2\u0129\u012e\5\36\20\2\u012a\u012b\7\7\2\2\u012b\u012d")
        buf.write("\5\36\20\2\u012c\u012a\3\2\2\2\u012d\u0130\3\2\2\2\u012e")
        buf.write("\u012c\3\2\2\2\u012e\u012f\3\2\2\2\u012f\u0132\3\2\2\2")
        buf.write("\u0130\u012e\3\2\2\2\u0131\u0133\7\7\2\2\u0132\u0131\3")
        buf.write("\2\2\2\u0132\u0133\3\2\2\2\u0133\35\3\2\2\2\u0134\u0135")
        buf.write("\5$\23\2\u0135\u013b\7\31\2\2\u0136\u0137\7\16\2\2\u0137")
        buf.write("\u0138\5\20\t\2\u0138\u0139\7\17\2\2\u0139\u013c\3\2\2")
        buf.write("\2\u013a\u013c\5\20\t\2\u013b\u0136\3\2\2\2\u013b\u013a")
        buf.write("\3\2\2\2\u013c\37\3\2\2\2\u013d\u013e\t\7\2\2\u013e!\3")
        buf.write("\2\2\2\u013f\u0140\7\b\2\2\u0140\u0145\5$\23\2\u0141\u0142")
        buf.write("\7\7\2\2\u0142\u0144\5$\23\2\u0143\u0141\3\2\2\2\u0144")
        buf.write("\u0147\3\2\2\2\u0145\u0143\3\2\2\2\u0145\u0146\3\2\2\2")
        buf.write("\u0146\u0148\3\2\2\2\u0147\u0145\3\2\2\2\u0148\u0149\7")
        buf.write("\t\2\2\u0149#\3\2\2\2\u014a\u0156\7\6\2\2\u014b\u014e")
        buf.write("\5\b\5\2\u014c\u014d\7\34\2\2\u014d\u014f\5\66\34\2\u014e")
        buf.write("\u014c\3\2\2\2\u014e\u014f\3\2\2\2\u014f\u0156\3\2\2\2")
        buf.write("\u0150\u0152\5\26\f\2\u0151\u0153\5\"\22\2\u0152\u0151")
        buf.write("\3\2\2\2\u0152\u0153\3\2\2\2\u0153\u0156\3\2\2\2\u0154")
        buf.write("\u0156\5\"\22\2\u0155\u014a\3\2\2\2\u0155\u014b\3\2\2")
        buf.write("\2\u0155\u0150\3\2\2\2\u0155\u0154\3\2\2\2\u0156%\3\2")
        buf.write("\2\2\u0157\u0159\5\26\f\2\u0158\u015a\5(\25\2\u0159\u0158")
        buf.write("\3\2\2\2\u0159\u015a\3\2\2\2\u015a\'\3\2\2\2\u015b\u015c")
        buf.write("\7\b\2\2\u015c\u0161\5*\26\2\u015d\u015e\7\7\2\2\u015e")
        buf.write("\u0160\5*\26\2\u015f\u015d\3\2\2\2\u0160\u0163\3\2\2\2")
        buf.write("\u0161\u015f\3\2\2\2\u0161\u0162\3\2\2\2\u0162\u0164\3")
        buf.write("\2\2\2\u0163\u0161\3\2\2\2\u0164\u0165\7\t\2\2\u0165)")
        buf.write("\3\2\2\2\u0166\u0169\5\b\5\2\u0167\u0169\5\26\f\2\u0168")
        buf.write("\u0166\3\2\2\2\u0168\u0167\3\2\2\2\u0169+\3\2\2\2\u016a")
        buf.write("\u0175\5.\30\2\u016b\u016c\5\60\31\2\u016c\u016d\7\7\2")
        buf.write("\2\u016d\u016f\3\2\2\2\u016e\u016b\3\2\2\2\u016f\u0172")
        buf.write("\3\2\2\2\u0170\u016e\3\2\2\2\u0170\u0171\3\2\2\2\u0171")
        buf.write("\u0173\3\2\2\2\u0172\u0170\3\2\2\2\u0173\u0175\5\62\32")
        buf.write("\2\u0174\u016a\3\2\2\2\u0174\u0170\3\2\2\2\u0175-\3\2")
        buf.write("\2\2\u0176\u017b\5\60\31\2\u0177\u0178\7\7\2\2\u0178\u017a")
        buf.write("\5\60\31\2\u0179\u0177\3\2\2\2\u017a\u017d\3\2\2\2\u017b")
        buf.write("\u0179\3\2\2\2\u017b\u017c\3\2\2\2\u017c\u017f\3\2\2\2")
        buf.write("\u017d\u017b\3\2\2\2\u017e\u0176\3\2\2\2\u017e\u017f\3")
        buf.write("\2\2\2\u017f/\3\2\2\2\u0180\u0183\5\b\5\2\u0181\u0182")
        buf.write("\7\34\2\2\u0182\u0184\5\66\34\2\u0183\u0181\3\2\2\2\u0183")
        buf.write("\u0184\3\2\2\2\u0184\61\3\2\2\2\u0185\u018a\5\64\33\2")
        buf.write("\u0186\u0187\7\7\2\2\u0187\u0189\5\64\33\2\u0188\u0186")
        buf.write("\3\2\2\2\u0189\u018c\3\2\2\2\u018a\u0188\3\2\2\2\u018a")
        buf.write("\u018b\3\2\2\2\u018b\63\3\2\2\2\u018c\u018a\3\2\2\2\u018d")
        buf.write("\u018e\7/\2\2\u018e\u018f\7\21\2\2\u018f\u0190\5\20\t")
        buf.write("\2\u0190\65\3\2\2\2\u0191\u0192\7\b\2\2\u0192\u01c5\7")
        buf.write("\t\2\2\u0193\u0194\7\b\2\2\u0194\u0195\5\66\34\2\u0195")
        buf.write("\u0196\7\t\2\2\u0196\u01c5\3\2\2\2\u0197\u0198\7\b\2\2")
        buf.write("\u0198\u0199\5\66\34\2\u0199\u019a\7\7\2\2\u019a\u019b")
        buf.write("\7\t\2\2\u019b\u01c5\3\2\2\2\u019c\u019d\7\b\2\2\u019d")
        buf.write("\u01a0\5\66\34\2\u019e\u019f\7\7\2\2\u019f\u01a1\5\66")
        buf.write("\34\2\u01a0\u019e\3\2\2\2\u01a1\u01a2\3\2\2\2\u01a2\u01a0")
        buf.write("\3\2\2\2\u01a2\u01a3\3\2\2\2\u01a3\u01a4\3\2\2\2\u01a4")
        buf.write("\u01a5\7\t\2\2\u01a5\u01c5\3\2\2\2\u01a6\u01a7\5\4\3\2")
        buf.write("\u01a7\u01a8\58\35\2\u01a8\u01c5\3\2\2\2\u01a9\u01c5\5")
        buf.write("\4\3\2\u01aa\u01ab\7\35\2\2\u01ab\u01ac\7\n\2\2\u01ac")
        buf.write("\u01ad\5:\36\2\u01ad\u01ae\7\7\2\2\u01ae\u01af\5\66\34")
        buf.write("\2\u01af\u01b0\7\13\2\2\u01b0\u01c5\3\2\2\2\u01b1\u01b3")
        buf.write("\7\24\2\2\u01b2\u01b4\58\35\2\u01b3\u01b2\3\2\2\2\u01b3")
        buf.write("\u01b4\3\2\2\2\u01b4\u01b5\3\2\2\2\u01b5\u01be\7\b\2\2")
        buf.write("\u01b6\u01bb\5\66\34\2\u01b7\u01b8\7\7\2\2\u01b8\u01ba")
        buf.write("\5\66\34\2\u01b9\u01b7\3\2\2\2\u01ba\u01bd\3\2\2\2\u01bb")
        buf.write("\u01b9\3\2\2\2\u01bb\u01bc\3\2\2\2\u01bc\u01bf\3\2\2\2")
        buf.write("\u01bd\u01bb\3\2\2\2\u01be\u01b6\3\2\2\2\u01be\u01bf\3")
        buf.write("\2\2\2\u01bf\u01c0\3\2\2\2\u01c0\u01c1\7\t\2\2\u01c1\u01c2")
        buf.write("\7\25\2\2\u01c2\u01c5\5\66\34\2\u01c3\u01c5\7\6\2\2\u01c4")
        buf.write("\u0191\3\2\2\2\u01c4\u0193\3\2\2\2\u01c4\u0197\3\2\2\2")
        buf.write("\u01c4\u019c\3\2\2\2\u01c4\u01a6\3\2\2\2\u01c4\u01a9\3")
        buf.write("\2\2\2\u01c4\u01aa\3\2\2\2\u01c4\u01b1\3\2\2\2\u01c4\u01c3")
        buf.write("\3\2\2\2\u01c5\67\3\2\2\2\u01c6\u01c7\7\n\2\2\u01c7\u01cc")
        buf.write("\5\66\34\2\u01c8\u01c9\7\7\2\2\u01c9\u01cb\5\66\34\2\u01ca")
        buf.write("\u01c8\3\2\2\2\u01cb\u01ce\3\2\2\2\u01cc\u01ca\3\2\2\2")
        buf.write("\u01cc\u01cd\3\2\2\2\u01cd\u01cf\3\2\2\2\u01ce\u01cc\3")
        buf.write("\2\2\2\u01cf\u01d0\7\13\2\2\u01d09\3\2\2\2\u01d1\u01d2")
        buf.write("\7\b\2\2\u01d2\u01df\7\t\2\2\u01d3\u01d4\7\b\2\2\u01d4")
        buf.write("\u01d7\5> \2\u01d5\u01d6\7\7\2\2\u01d6\u01d8\5> \2\u01d7")
        buf.write("\u01d5\3\2\2\2\u01d8\u01d9\3\2\2\2\u01d9\u01d7\3\2\2\2")
        buf.write("\u01d9\u01da\3\2\2\2\u01da\u01db\3\2\2\2\u01db\u01dc\7")
        buf.write("\t\2\2\u01dc\u01df\3\2\2\2\u01dd\u01df\5> \2\u01de\u01d1")
        buf.write("\3\2\2\2\u01de\u01d3\3\2\2\2\u01de\u01dd\3\2\2\2\u01df")
        buf.write(";\3\2\2\2\u01e0\u01e1\7\36\2\2\u01e1\u01e2\7\n\2\2\u01e2")
        buf.write("\u01e3\7/\2\2\u01e3\u01e4\7\13\2\2\u01e4\u01e5\7\n\2\2")
        buf.write("\u01e5\u01e6\7\61\2\2\u01e6\u01e7\7\13\2\2\u01e7=\3\2")
        buf.write("\2\2\u01e8\u01ef\5<\37\2\u01e9\u01ea\7\b\2\2\u01ea\u01eb")
        buf.write("\5> \2\u01eb\u01ec\7\t\2\2\u01ec\u01ef\3\2\2\2\u01ed\u01ef")
        buf.write("\7\61\2\2\u01ee\u01e8\3\2\2\2\u01ee\u01e9\3\2\2\2\u01ee")
        buf.write("\u01ed\3\2\2\2\u01ef?\3\2\2\2\u01f0\u01f1\7\16\2\2\u01f1")
        buf.write("\u01f2\5\20\t\2\u01f2\u01f3\7\17\2\2\u01f3A\3\2\2\2\u01f4")
        buf.write("\u01f8\7\60\2\2\u01f5\u01f8\7\61\2\2\u01f6\u01f8\7.\2")
        buf.write("\2\u01f7\u01f4\3\2\2\2\u01f7\u01f5\3\2\2\2\u01f7\u01f6")
        buf.write("\3\2\2\2\u01f8C\3\2\2\2\u01f9\u01fe\5\4\3\2\u01fa\u01fe")
        buf.write("\5\6\4\2\u01fb\u01fe\5\b\5\2\u01fc\u01fe\5\n\6\2\u01fd")
        buf.write("\u01f9\3\2\2\2\u01fd\u01fa\3\2\2\2\u01fd\u01fb\3\2\2\2")
        buf.write("\u01fd\u01fc\3\2\2\2\u01feE\3\2\2\28JNQZknvz\u0091\u009b")
        buf.write("\u009e\u00ad\u00c2\u00db\u00dd\u00e2\u00e9\u00f0\u00f7")
        buf.write("\u00ff\u0104\u0108\u010c\u0115\u0119\u0122\u0127\u012e")
        buf.write("\u0132\u013b\u0145\u014e\u0152\u0155\u0159\u0161\u0168")
        buf.write("\u0170\u0174\u017b\u017e\u0183\u018a\u01a2\u01b3\u01bb")
        buf.write("\u01be\u01c4\u01cc\u01d9\u01de\u01ee\u01f7\u01fd")
        return buf.getvalue()


class RelayParser ( Parser ):

    grammarFileName = "Relay.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'.'", "'@'", "'%'", "'_'", "','", "'('", 
                     "')'", "'['", "']'", "'if'", "'else'", "'{'", "'}'", 
                     "'let'", "'='", "';'", "';;'", "'fn'", "'->'", "'def'", 
                     "'extern'", "'type'", "'=>'", "'match'", "'match?'", 
                     "':'", "'Tensor'", "'meta'", "'v0.0.4'", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "'*'", "'/'", 
                     "'+'", "'-'", "'<'", "'>'", "'<='", "'>='", "'=='", 
                     "'!='" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "SEMVER", "COMMENT", "WS", "LINE_COMMENT", 
                      "QUOTED_STRING", "MUL", "DIV", "ADD", "SUB", "LT", 
                      "GT", "LE", "GE", "EQ", "NE", "BOOL_LIT", "CNAME", 
                      "FLOAT", "NAT", "METADATA" ]

    RULE_prog = 0
    RULE_generalIdent = 1
    RULE_globalVar = 2
    RULE_localVar = 3
    RULE_graphVar = 4
    RULE_exprList = 5
    RULE_callList = 6
    RULE_expr = 7
    RULE_func = 8
    RULE_defn = 9
    RULE_constructorName = 10
    RULE_adtConsDefnList = 11
    RULE_adtConsDefn = 12
    RULE_matchClauseList = 13
    RULE_matchClause = 14
    RULE_matchType = 15
    RULE_patternList = 16
    RULE_pattern = 17
    RULE_adtCons = 18
    RULE_adtConsParamList = 19
    RULE_adtConsParam = 20
    RULE_argList = 21
    RULE_varList = 22
    RULE_var = 23
    RULE_attrSeq = 24
    RULE_attr = 25
    RULE_typeExpr = 26
    RULE_typeParamList = 27
    RULE_shapeList = 28
    RULE_meta = 29
    RULE_shape = 30
    RULE_body = 31
    RULE_scalar = 32
    RULE_ident = 33

    ruleNames =  [ "prog", "generalIdent", "globalVar", "localVar", "graphVar", 
                   "exprList", "callList", "expr", "func", "defn", "constructorName", 
                   "adtConsDefnList", "adtConsDefn", "matchClauseList", 
                   "matchClause", "matchType", "patternList", "pattern", 
                   "adtCons", "adtConsParamList", "adtConsParam", "argList", 
                   "varList", "var", "attrSeq", "attr", "typeExpr", "typeParamList", 
                   "shapeList", "meta", "shape", "body", "scalar", "ident" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    T__26=27
    T__27=28
    SEMVER=29
    COMMENT=30
    WS=31
    LINE_COMMENT=32
    QUOTED_STRING=33
    MUL=34
    DIV=35
    ADD=36
    SUB=37
    LT=38
    GT=39
    LE=40
    GE=41
    EQ=42
    NE=43
    BOOL_LIT=44
    CNAME=45
    FLOAT=46
    NAT=47
    METADATA=48

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SEMVER(self):
            return self.getToken(RelayParser.SEMVER, 0)

        def EOF(self):
            return self.getToken(RelayParser.EOF, 0)

        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)


        def METADATA(self):
            return self.getToken(RelayParser.METADATA, 0)

        def defn(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.DefnContext)
            else:
                return self.getTypedRuleContext(RelayParser.DefnContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_prog

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProg" ):
                return visitor.visitProg(self)
            else:
                return visitor.visitChildren(self)




    def prog(self):

        localctx = RelayParser.ProgContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_prog)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 68
            self.match(RelayParser.SEMVER)
            self.state = 76
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.EOF, RelayParser.T__19, RelayParser.T__20, RelayParser.T__21, RelayParser.METADATA]:
                self.state = 72
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.T__19) | (1 << RelayParser.T__20) | (1 << RelayParser.T__21))) != 0):
                    self.state = 69
                    self.defn()
                    self.state = 74
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                pass
            elif token in [RelayParser.T__1, RelayParser.T__2, RelayParser.T__5, RelayParser.T__7, RelayParser.T__9, RelayParser.T__13, RelayParser.T__17, RelayParser.T__23, RelayParser.T__24, RelayParser.T__27, RelayParser.QUOTED_STRING, RelayParser.SUB, RelayParser.BOOL_LIT, RelayParser.CNAME, RelayParser.FLOAT, RelayParser.NAT]:
                self.state = 75
                self.expr(0)
                pass
            else:
                raise NoViableAltException(self)

            self.state = 79
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.METADATA:
                self.state = 78
                self.match(RelayParser.METADATA)


            self.state = 81
            self.match(RelayParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GeneralIdentContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CNAME(self, i:int=None):
            if i is None:
                return self.getTokens(RelayParser.CNAME)
            else:
                return self.getToken(RelayParser.CNAME, i)

        def getRuleIndex(self):
            return RelayParser.RULE_generalIdent

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGeneralIdent" ):
                return visitor.visitGeneralIdent(self)
            else:
                return visitor.visitChildren(self)




    def generalIdent(self):

        localctx = RelayParser.GeneralIdentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_generalIdent)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 83
            self.match(RelayParser.CNAME)
            self.state = 88
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,3,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 84
                    self.match(RelayParser.T__0)
                    self.state = 85
                    self.match(RelayParser.CNAME) 
                self.state = 90
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,3,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GlobalVarContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CNAME(self):
            return self.getToken(RelayParser.CNAME, 0)

        def getRuleIndex(self):
            return RelayParser.RULE_globalVar

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGlobalVar" ):
                return visitor.visitGlobalVar(self)
            else:
                return visitor.visitChildren(self)




    def globalVar(self):

        localctx = RelayParser.GlobalVarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_globalVar)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 91
            self.match(RelayParser.T__1)
            self.state = 92
            self.match(RelayParser.CNAME)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LocalVarContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CNAME(self):
            return self.getToken(RelayParser.CNAME, 0)

        def getRuleIndex(self):
            return RelayParser.RULE_localVar

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLocalVar" ):
                return visitor.visitLocalVar(self)
            else:
                return visitor.visitChildren(self)




    def localVar(self):

        localctx = RelayParser.LocalVarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_localVar)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 94
            self.match(RelayParser.T__2)
            self.state = 95
            _la = self._input.LA(1)
            if not(_la==RelayParser.T__3 or _la==RelayParser.CNAME):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GraphVarContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NAT(self):
            return self.getToken(RelayParser.NAT, 0)

        def getRuleIndex(self):
            return RelayParser.RULE_graphVar

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGraphVar" ):
                return visitor.visitGraphVar(self)
            else:
                return visitor.visitChildren(self)




    def graphVar(self):

        localctx = RelayParser.GraphVarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_graphVar)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 97
            self.match(RelayParser.T__2)
            self.state = 98
            self.match(RelayParser.NAT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.ExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.ExprContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_exprList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExprList" ):
                return visitor.visitExprList(self)
            else:
                return visitor.visitChildren(self)




    def exprList(self):

        localctx = RelayParser.ExprListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_exprList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 108
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.T__1) | (1 << RelayParser.T__2) | (1 << RelayParser.T__5) | (1 << RelayParser.T__7) | (1 << RelayParser.T__9) | (1 << RelayParser.T__13) | (1 << RelayParser.T__17) | (1 << RelayParser.T__23) | (1 << RelayParser.T__24) | (1 << RelayParser.T__27) | (1 << RelayParser.QUOTED_STRING) | (1 << RelayParser.SUB) | (1 << RelayParser.BOOL_LIT) | (1 << RelayParser.CNAME) | (1 << RelayParser.FLOAT) | (1 << RelayParser.NAT))) != 0):
                self.state = 100
                self.expr(0)
                self.state = 105
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==RelayParser.T__4:
                    self.state = 101
                    self.match(RelayParser.T__4)
                    self.state = 102
                    self.expr(0)
                    self.state = 107
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)



        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CallListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_callList

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class CallWithAttrContext(CallListContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.CallListContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def attrSeq(self):
            return self.getTypedRuleContext(RelayParser.AttrSeqContext,0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.ExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.ExprContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCallWithAttr" ):
                return visitor.visitCallWithAttr(self)
            else:
                return visitor.visitChildren(self)


    class CallNoAttrContext(CallListContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.CallListContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def exprList(self):
            return self.getTypedRuleContext(RelayParser.ExprListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCallNoAttr" ):
                return visitor.visitCallNoAttr(self)
            else:
                return visitor.visitChildren(self)



    def callList(self):

        localctx = RelayParser.CallListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_callList)
        try:
            self.state = 120
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                localctx = RelayParser.CallNoAttrContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 110
                self.exprList()
                pass

            elif la_ == 2:
                localctx = RelayParser.CallWithAttrContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 116
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 111
                        self.expr(0)
                        self.state = 112
                        self.match(RelayParser.T__4) 
                    self.state = 118
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,6,self._ctx)

                self.state = 119
                self.attrSeq()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_expr

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class FuncExprContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def func(self):
            return self.getTypedRuleContext(RelayParser.FuncContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFuncExpr" ):
                return visitor.visitFuncExpr(self)
            else:
                return visitor.visitChildren(self)


    class MetaExprContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def meta(self):
            return self.getTypedRuleContext(RelayParser.MetaContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMetaExpr" ):
                return visitor.visitMetaExpr(self)
            else:
                return visitor.visitChildren(self)


    class MatchContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def matchType(self):
            return self.getTypedRuleContext(RelayParser.MatchTypeContext,0)

        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)

        def matchClauseList(self):
            return self.getTypedRuleContext(RelayParser.MatchClauseListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatch" ):
                return visitor.visitMatch(self)
            else:
                return visitor.visitChildren(self)


    class TensorContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.ExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.ExprContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTensor" ):
                return visitor.visitTensor(self)
            else:
                return visitor.visitChildren(self)


    class GraphContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def graphVar(self):
            return self.getTypedRuleContext(RelayParser.GraphVarContext,0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.ExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.ExprContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGraph" ):
                return visitor.visitGraph(self)
            else:
                return visitor.visitChildren(self)


    class IdentExprContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ident(self):
            return self.getTypedRuleContext(RelayParser.IdentContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIdentExpr" ):
                return visitor.visitIdentExpr(self)
            else:
                return visitor.visitChildren(self)


    class StringExprContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def QUOTED_STRING(self):
            return self.getToken(RelayParser.QUOTED_STRING, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStringExpr" ):
                return visitor.visitStringExpr(self)
            else:
                return visitor.visitChildren(self)


    class CallContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)

        def callList(self):
            return self.getTypedRuleContext(RelayParser.CallListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCall" ):
                return visitor.visitCall(self)
            else:
                return visitor.visitChildren(self)


    class NegContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def SUB(self):
            return self.getToken(RelayParser.SUB, 0)
        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNeg" ):
                return visitor.visitNeg(self)
            else:
                return visitor.visitChildren(self)


    class TupleContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.ExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.ExprContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTuple" ):
                return visitor.visitTuple(self)
            else:
                return visitor.visitChildren(self)


    class ParenContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitParen" ):
                return visitor.visitParen(self)
            else:
                return visitor.visitChildren(self)


    class ScalarExprContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def scalar(self):
            return self.getTypedRuleContext(RelayParser.ScalarContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitScalarExpr" ):
                return visitor.visitScalarExpr(self)
            else:
                return visitor.visitChildren(self)


    class LetContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def var(self):
            return self.getTypedRuleContext(RelayParser.VarContext,0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.ExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.ExprContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLet" ):
                return visitor.visitLet(self)
            else:
                return visitor.visitChildren(self)


    class ProjectionContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)

        def NAT(self):
            return self.getToken(RelayParser.NAT, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProjection" ):
                return visitor.visitProjection(self)
            else:
                return visitor.visitChildren(self)


    class IfElseContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)

        def body(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.BodyContext)
            else:
                return self.getTypedRuleContext(RelayParser.BodyContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIfElse" ):
                return visitor.visitIfElse(self)
            else:
                return visitor.visitChildren(self)


    class BinOpContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ExprContext
            super().__init__(parser)
            self.op = None # Token
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.ExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.ExprContext,i)

        def MUL(self):
            return self.getToken(RelayParser.MUL, 0)
        def DIV(self):
            return self.getToken(RelayParser.DIV, 0)
        def ADD(self):
            return self.getToken(RelayParser.ADD, 0)
        def SUB(self):
            return self.getToken(RelayParser.SUB, 0)
        def LT(self):
            return self.getToken(RelayParser.LT, 0)
        def GT(self):
            return self.getToken(RelayParser.GT, 0)
        def LE(self):
            return self.getToken(RelayParser.LE, 0)
        def GE(self):
            return self.getToken(RelayParser.GE, 0)
        def EQ(self):
            return self.getToken(RelayParser.EQ, 0)
        def NE(self):
            return self.getToken(RelayParser.NE, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBinOp" ):
                return visitor.visitBinOp(self)
            else:
                return visitor.visitChildren(self)



    def expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = RelayParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 14
        self.enterRecursionRule(localctx, 14, self.RULE_expr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 192
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
            if la_ == 1:
                localctx = RelayParser.ParenContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 123
                self.match(RelayParser.T__5)
                self.state = 124
                self.expr(0)
                self.state = 125
                self.match(RelayParser.T__6)
                pass

            elif la_ == 2:
                localctx = RelayParser.NegContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 127
                self.match(RelayParser.SUB)
                self.state = 128
                self.expr(20)
                pass

            elif la_ == 3:
                localctx = RelayParser.FuncExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 129
                self.func()
                pass

            elif la_ == 4:
                localctx = RelayParser.TupleContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 130
                self.match(RelayParser.T__5)
                self.state = 131
                self.match(RelayParser.T__6)
                pass

            elif la_ == 5:
                localctx = RelayParser.TupleContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 132
                self.match(RelayParser.T__5)
                self.state = 133
                self.expr(0)
                self.state = 134
                self.match(RelayParser.T__4)
                self.state = 135
                self.match(RelayParser.T__6)
                pass

            elif la_ == 6:
                localctx = RelayParser.TupleContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 137
                self.match(RelayParser.T__5)
                self.state = 138
                self.expr(0)
                self.state = 141 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 139
                    self.match(RelayParser.T__4)
                    self.state = 140
                    self.expr(0)
                    self.state = 143 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (_la==RelayParser.T__4):
                        break

                self.state = 145
                self.match(RelayParser.T__6)
                pass

            elif la_ == 7:
                localctx = RelayParser.TensorContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 147
                self.match(RelayParser.T__7)
                self.state = 156
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.T__1) | (1 << RelayParser.T__2) | (1 << RelayParser.T__5) | (1 << RelayParser.T__7) | (1 << RelayParser.T__9) | (1 << RelayParser.T__13) | (1 << RelayParser.T__17) | (1 << RelayParser.T__23) | (1 << RelayParser.T__24) | (1 << RelayParser.T__27) | (1 << RelayParser.QUOTED_STRING) | (1 << RelayParser.SUB) | (1 << RelayParser.BOOL_LIT) | (1 << RelayParser.CNAME) | (1 << RelayParser.FLOAT) | (1 << RelayParser.NAT))) != 0):
                    self.state = 148
                    self.expr(0)
                    self.state = 153
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la==RelayParser.T__4:
                        self.state = 149
                        self.match(RelayParser.T__4)
                        self.state = 150
                        self.expr(0)
                        self.state = 155
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)



                self.state = 158
                self.match(RelayParser.T__8)
                pass

            elif la_ == 8:
                localctx = RelayParser.IfElseContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 159
                self.match(RelayParser.T__9)
                self.state = 160
                self.match(RelayParser.T__5)
                self.state = 161
                self.expr(0)
                self.state = 162
                self.match(RelayParser.T__6)
                self.state = 163
                self.body()
                self.state = 164
                self.match(RelayParser.T__10)
                self.state = 165
                self.body()
                pass

            elif la_ == 9:
                localctx = RelayParser.MatchContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 167
                self.matchType()
                self.state = 168
                self.expr(0)
                self.state = 169
                self.match(RelayParser.T__11)
                self.state = 171
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.T__2) | (1 << RelayParser.T__3) | (1 << RelayParser.T__5) | (1 << RelayParser.CNAME))) != 0):
                    self.state = 170
                    self.matchClauseList()


                self.state = 173
                self.match(RelayParser.T__12)
                pass

            elif la_ == 10:
                localctx = RelayParser.LetContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 175
                self.match(RelayParser.T__13)
                self.state = 176
                self.var()
                self.state = 177
                self.match(RelayParser.T__14)
                self.state = 178
                self.expr(0)
                self.state = 179
                self.match(RelayParser.T__15)
                self.state = 180
                self.expr(7)
                pass

            elif la_ == 11:
                localctx = RelayParser.GraphContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 182
                self.graphVar()
                self.state = 183
                self.match(RelayParser.T__14)
                self.state = 184
                self.expr(0)
                self.state = 185
                self.match(RelayParser.T__15)
                self.state = 186
                self.expr(5)
                pass

            elif la_ == 12:
                localctx = RelayParser.IdentExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 188
                self.ident()
                pass

            elif la_ == 13:
                localctx = RelayParser.ScalarExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 189
                self.scalar()
                pass

            elif la_ == 14:
                localctx = RelayParser.MetaExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 190
                self.meta()
                pass

            elif la_ == 15:
                localctx = RelayParser.StringExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 191
                self.match(RelayParser.QUOTED_STRING)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 219
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,14,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 217
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
                    if la_ == 1:
                        localctx = RelayParser.BinOpContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 194
                        if not self.precpred(self._ctx, 19):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 19)")
                        self.state = 195
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not(_la==RelayParser.MUL or _la==RelayParser.DIV):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 196
                        self.expr(20)
                        pass

                    elif la_ == 2:
                        localctx = RelayParser.BinOpContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 197
                        if not self.precpred(self._ctx, 18):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 18)")
                        self.state = 198
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not(_la==RelayParser.ADD or _la==RelayParser.SUB):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 199
                        self.expr(19)
                        pass

                    elif la_ == 3:
                        localctx = RelayParser.BinOpContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 200
                        if not self.precpred(self._ctx, 17):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 17)")
                        self.state = 201
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.LT) | (1 << RelayParser.GT) | (1 << RelayParser.LE) | (1 << RelayParser.GE))) != 0)):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 202
                        self.expr(18)
                        pass

                    elif la_ == 4:
                        localctx = RelayParser.BinOpContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 203
                        if not self.precpred(self._ctx, 16):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 16)")
                        self.state = 204
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not(_la==RelayParser.EQ or _la==RelayParser.NE):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 205
                        self.expr(17)
                        pass

                    elif la_ == 5:
                        localctx = RelayParser.LetContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 206
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 207
                        self.match(RelayParser.T__16)
                        self.state = 208
                        self.expr(7)
                        pass

                    elif la_ == 6:
                        localctx = RelayParser.CallContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 209
                        if not self.precpred(self._ctx, 21):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 21)")
                        self.state = 210
                        self.match(RelayParser.T__5)
                        self.state = 211
                        self.callList()
                        self.state = 212
                        self.match(RelayParser.T__6)
                        pass

                    elif la_ == 7:
                        localctx = RelayParser.ProjectionContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 214
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 215
                        self.match(RelayParser.T__0)
                        self.state = 216
                        self.match(RelayParser.NAT)
                        pass

             
                self.state = 221
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,14,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class FuncContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def argList(self):
            return self.getTypedRuleContext(RelayParser.ArgListContext,0)


        def body(self):
            return self.getTypedRuleContext(RelayParser.BodyContext,0)


        def typeParamList(self):
            return self.getTypedRuleContext(RelayParser.TypeParamListContext,0)


        def typeExpr(self):
            return self.getTypedRuleContext(RelayParser.TypeExprContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_func

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFunc" ):
                return visitor.visitFunc(self)
            else:
                return visitor.visitChildren(self)




    def func(self):

        localctx = RelayParser.FuncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_func)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 222
            self.match(RelayParser.T__17)
            self.state = 224
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__7:
                self.state = 223
                self.typeParamList()


            self.state = 226
            self.match(RelayParser.T__5)
            self.state = 227
            self.argList()
            self.state = 228
            self.match(RelayParser.T__6)
            self.state = 231
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__18:
                self.state = 229
                self.match(RelayParser.T__18)
                self.state = 230
                self.typeExpr()


            self.state = 233
            self.body()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DefnContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_defn

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class ExternAdtDefnContext(DefnContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.DefnContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def generalIdent(self):
            return self.getTypedRuleContext(RelayParser.GeneralIdentContext,0)

        def typeParamList(self):
            return self.getTypedRuleContext(RelayParser.TypeParamListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExternAdtDefn" ):
                return visitor.visitExternAdtDefn(self)
            else:
                return visitor.visitChildren(self)


    class FuncDefnContext(DefnContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.DefnContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def globalVar(self):
            return self.getTypedRuleContext(RelayParser.GlobalVarContext,0)

        def argList(self):
            return self.getTypedRuleContext(RelayParser.ArgListContext,0)

        def body(self):
            return self.getTypedRuleContext(RelayParser.BodyContext,0)

        def typeParamList(self):
            return self.getTypedRuleContext(RelayParser.TypeParamListContext,0)

        def typeExpr(self):
            return self.getTypedRuleContext(RelayParser.TypeExprContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFuncDefn" ):
                return visitor.visitFuncDefn(self)
            else:
                return visitor.visitChildren(self)


    class AdtDefnContext(DefnContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.DefnContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def generalIdent(self):
            return self.getTypedRuleContext(RelayParser.GeneralIdentContext,0)

        def typeParamList(self):
            return self.getTypedRuleContext(RelayParser.TypeParamListContext,0)

        def adtConsDefnList(self):
            return self.getTypedRuleContext(RelayParser.AdtConsDefnListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdtDefn" ):
                return visitor.visitAdtDefn(self)
            else:
                return visitor.visitChildren(self)



    def defn(self):

        localctx = RelayParser.DefnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_defn)
        self._la = 0 # Token type
        try:
            self.state = 266
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__19]:
                localctx = RelayParser.FuncDefnContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 235
                self.match(RelayParser.T__19)
                self.state = 236
                self.globalVar()
                self.state = 238
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__7:
                    self.state = 237
                    self.typeParamList()


                self.state = 240
                self.match(RelayParser.T__5)
                self.state = 241
                self.argList()
                self.state = 242
                self.match(RelayParser.T__6)
                self.state = 245
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__18:
                    self.state = 243
                    self.match(RelayParser.T__18)
                    self.state = 244
                    self.typeExpr()


                self.state = 247
                self.body()
                pass
            elif token in [RelayParser.T__20]:
                localctx = RelayParser.ExternAdtDefnContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 249
                self.match(RelayParser.T__20)
                self.state = 250
                self.match(RelayParser.T__21)
                self.state = 251
                self.generalIdent()
                self.state = 253
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__7:
                    self.state = 252
                    self.typeParamList()


                pass
            elif token in [RelayParser.T__21]:
                localctx = RelayParser.AdtDefnContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 255
                self.match(RelayParser.T__21)
                self.state = 256
                self.generalIdent()
                self.state = 258
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__7:
                    self.state = 257
                    self.typeParamList()


                self.state = 260
                self.match(RelayParser.T__11)
                self.state = 262
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.CNAME:
                    self.state = 261
                    self.adtConsDefnList()


                self.state = 264
                self.match(RelayParser.T__12)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConstructorNameContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CNAME(self):
            return self.getToken(RelayParser.CNAME, 0)

        def getRuleIndex(self):
            return RelayParser.RULE_constructorName

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConstructorName" ):
                return visitor.visitConstructorName(self)
            else:
                return visitor.visitChildren(self)




    def constructorName(self):

        localctx = RelayParser.ConstructorNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_constructorName)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 268
            self.match(RelayParser.CNAME)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AdtConsDefnListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def adtConsDefn(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.AdtConsDefnContext)
            else:
                return self.getTypedRuleContext(RelayParser.AdtConsDefnContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_adtConsDefnList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdtConsDefnList" ):
                return visitor.visitAdtConsDefnList(self)
            else:
                return visitor.visitChildren(self)




    def adtConsDefnList(self):

        localctx = RelayParser.AdtConsDefnListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_adtConsDefnList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 270
            self.adtConsDefn()
            self.state = 275
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,23,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 271
                    self.match(RelayParser.T__4)
                    self.state = 272
                    self.adtConsDefn() 
                self.state = 277
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,23,self._ctx)

            self.state = 279
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__4:
                self.state = 278
                self.match(RelayParser.T__4)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AdtConsDefnContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def constructorName(self):
            return self.getTypedRuleContext(RelayParser.ConstructorNameContext,0)


        def typeExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.TypeExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.TypeExprContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_adtConsDefn

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdtConsDefn" ):
                return visitor.visitAdtConsDefn(self)
            else:
                return visitor.visitChildren(self)




    def adtConsDefn(self):

        localctx = RelayParser.AdtConsDefnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_adtConsDefn)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 281
            self.constructorName()
            self.state = 293
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__5:
                self.state = 282
                self.match(RelayParser.T__5)
                self.state = 283
                self.typeExpr()
                self.state = 288
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==RelayParser.T__4:
                    self.state = 284
                    self.match(RelayParser.T__4)
                    self.state = 285
                    self.typeExpr()
                    self.state = 290
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 291
                self.match(RelayParser.T__6)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MatchClauseListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def matchClause(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.MatchClauseContext)
            else:
                return self.getTypedRuleContext(RelayParser.MatchClauseContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_matchClauseList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatchClauseList" ):
                return visitor.visitMatchClauseList(self)
            else:
                return visitor.visitChildren(self)




    def matchClauseList(self):

        localctx = RelayParser.MatchClauseListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_matchClauseList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 295
            self.matchClause()
            self.state = 300
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,27,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 296
                    self.match(RelayParser.T__4)
                    self.state = 297
                    self.matchClause() 
                self.state = 302
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,27,self._ctx)

            self.state = 304
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__4:
                self.state = 303
                self.match(RelayParser.T__4)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MatchClauseContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def pattern(self):
            return self.getTypedRuleContext(RelayParser.PatternContext,0)


        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_matchClause

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatchClause" ):
                return visitor.visitMatchClause(self)
            else:
                return visitor.visitChildren(self)




    def matchClause(self):

        localctx = RelayParser.MatchClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_matchClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 306
            self.pattern()
            self.state = 307
            self.match(RelayParser.T__22)
            self.state = 313
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__11]:
                self.state = 308
                self.match(RelayParser.T__11)
                self.state = 309
                self.expr(0)
                self.state = 310
                self.match(RelayParser.T__12)
                pass
            elif token in [RelayParser.T__1, RelayParser.T__2, RelayParser.T__5, RelayParser.T__7, RelayParser.T__9, RelayParser.T__13, RelayParser.T__17, RelayParser.T__23, RelayParser.T__24, RelayParser.T__27, RelayParser.QUOTED_STRING, RelayParser.SUB, RelayParser.BOOL_LIT, RelayParser.CNAME, RelayParser.FLOAT, RelayParser.NAT]:
                self.state = 312
                self.expr(0)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MatchTypeContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_matchType

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatchType" ):
                return visitor.visitMatchType(self)
            else:
                return visitor.visitChildren(self)




    def matchType(self):

        localctx = RelayParser.MatchTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_matchType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 315
            _la = self._input.LA(1)
            if not(_la==RelayParser.T__23 or _la==RelayParser.T__24):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PatternListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def pattern(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.PatternContext)
            else:
                return self.getTypedRuleContext(RelayParser.PatternContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_patternList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPatternList" ):
                return visitor.visitPatternList(self)
            else:
                return visitor.visitChildren(self)




    def patternList(self):

        localctx = RelayParser.PatternListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_patternList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 317
            self.match(RelayParser.T__5)
            self.state = 318
            self.pattern()
            self.state = 323
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==RelayParser.T__4:
                self.state = 319
                self.match(RelayParser.T__4)
                self.state = 320
                self.pattern()
                self.state = 325
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 326
            self.match(RelayParser.T__6)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PatternContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_pattern

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class WildcardPatternContext(PatternContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.PatternContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWildcardPattern" ):
                return visitor.visitWildcardPattern(self)
            else:
                return visitor.visitChildren(self)


    class ConstructorPatternContext(PatternContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.PatternContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def constructorName(self):
            return self.getTypedRuleContext(RelayParser.ConstructorNameContext,0)

        def patternList(self):
            return self.getTypedRuleContext(RelayParser.PatternListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConstructorPattern" ):
                return visitor.visitConstructorPattern(self)
            else:
                return visitor.visitChildren(self)


    class TuplePatternContext(PatternContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.PatternContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def patternList(self):
            return self.getTypedRuleContext(RelayParser.PatternListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTuplePattern" ):
                return visitor.visitTuplePattern(self)
            else:
                return visitor.visitChildren(self)


    class VarPatternContext(PatternContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.PatternContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def localVar(self):
            return self.getTypedRuleContext(RelayParser.LocalVarContext,0)

        def typeExpr(self):
            return self.getTypedRuleContext(RelayParser.TypeExprContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVarPattern" ):
                return visitor.visitVarPattern(self)
            else:
                return visitor.visitChildren(self)



    def pattern(self):

        localctx = RelayParser.PatternContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_pattern)
        self._la = 0 # Token type
        try:
            self.state = 339
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__3]:
                localctx = RelayParser.WildcardPatternContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 328
                self.match(RelayParser.T__3)
                pass
            elif token in [RelayParser.T__2]:
                localctx = RelayParser.VarPatternContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 329
                self.localVar()
                self.state = 332
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__25:
                    self.state = 330
                    self.match(RelayParser.T__25)
                    self.state = 331
                    self.typeExpr()


                pass
            elif token in [RelayParser.CNAME]:
                localctx = RelayParser.ConstructorPatternContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 334
                self.constructorName()
                self.state = 336
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__5:
                    self.state = 335
                    self.patternList()


                pass
            elif token in [RelayParser.T__5]:
                localctx = RelayParser.TuplePatternContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 338
                self.patternList()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AdtConsContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def constructorName(self):
            return self.getTypedRuleContext(RelayParser.ConstructorNameContext,0)


        def adtConsParamList(self):
            return self.getTypedRuleContext(RelayParser.AdtConsParamListContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_adtCons

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdtCons" ):
                return visitor.visitAdtCons(self)
            else:
                return visitor.visitChildren(self)




    def adtCons(self):

        localctx = RelayParser.AdtConsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_adtCons)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 341
            self.constructorName()
            self.state = 343
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__5:
                self.state = 342
                self.adtConsParamList()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AdtConsParamListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def adtConsParam(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.AdtConsParamContext)
            else:
                return self.getTypedRuleContext(RelayParser.AdtConsParamContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_adtConsParamList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdtConsParamList" ):
                return visitor.visitAdtConsParamList(self)
            else:
                return visitor.visitChildren(self)




    def adtConsParamList(self):

        localctx = RelayParser.AdtConsParamListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_adtConsParamList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 345
            self.match(RelayParser.T__5)
            self.state = 346
            self.adtConsParam()
            self.state = 351
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==RelayParser.T__4:
                self.state = 347
                self.match(RelayParser.T__4)
                self.state = 348
                self.adtConsParam()
                self.state = 353
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 354
            self.match(RelayParser.T__6)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AdtConsParamContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def localVar(self):
            return self.getTypedRuleContext(RelayParser.LocalVarContext,0)


        def constructorName(self):
            return self.getTypedRuleContext(RelayParser.ConstructorNameContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_adtConsParam

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdtConsParam" ):
                return visitor.visitAdtConsParam(self)
            else:
                return visitor.visitChildren(self)




    def adtConsParam(self):

        localctx = RelayParser.AdtConsParamContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_adtConsParam)
        try:
            self.state = 358
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__2]:
                self.enterOuterAlt(localctx, 1)
                self.state = 356
                self.localVar()
                pass
            elif token in [RelayParser.CNAME]:
                self.enterOuterAlt(localctx, 2)
                self.state = 357
                self.constructorName()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ArgListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_argList

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class ArgNoAttrContext(ArgListContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ArgListContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def varList(self):
            return self.getTypedRuleContext(RelayParser.VarListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitArgNoAttr" ):
                return visitor.visitArgNoAttr(self)
            else:
                return visitor.visitChildren(self)


    class ArgWithAttrContext(ArgListContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ArgListContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def attrSeq(self):
            return self.getTypedRuleContext(RelayParser.AttrSeqContext,0)

        def var(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.VarContext)
            else:
                return self.getTypedRuleContext(RelayParser.VarContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitArgWithAttr" ):
                return visitor.visitArgWithAttr(self)
            else:
                return visitor.visitChildren(self)



    def argList(self):

        localctx = RelayParser.ArgListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_argList)
        self._la = 0 # Token type
        try:
            self.state = 370
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,38,self._ctx)
            if la_ == 1:
                localctx = RelayParser.ArgNoAttrContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 360
                self.varList()
                pass

            elif la_ == 2:
                localctx = RelayParser.ArgWithAttrContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 366
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==RelayParser.T__2:
                    self.state = 361
                    self.var()
                    self.state = 362
                    self.match(RelayParser.T__4)
                    self.state = 368
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 369
                self.attrSeq()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VarListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def var(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.VarContext)
            else:
                return self.getTypedRuleContext(RelayParser.VarContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_varList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVarList" ):
                return visitor.visitVarList(self)
            else:
                return visitor.visitChildren(self)




    def varList(self):

        localctx = RelayParser.VarListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_varList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 380
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__2:
                self.state = 372
                self.var()
                self.state = 377
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==RelayParser.T__4:
                    self.state = 373
                    self.match(RelayParser.T__4)
                    self.state = 374
                    self.var()
                    self.state = 379
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)



        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VarContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def localVar(self):
            return self.getTypedRuleContext(RelayParser.LocalVarContext,0)


        def typeExpr(self):
            return self.getTypedRuleContext(RelayParser.TypeExprContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_var

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVar" ):
                return visitor.visitVar(self)
            else:
                return visitor.visitChildren(self)




    def var(self):

        localctx = RelayParser.VarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_var)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 382
            self.localVar()
            self.state = 385
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__25:
                self.state = 383
                self.match(RelayParser.T__25)
                self.state = 384
                self.typeExpr()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AttrSeqContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def attr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.AttrContext)
            else:
                return self.getTypedRuleContext(RelayParser.AttrContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_attrSeq

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAttrSeq" ):
                return visitor.visitAttrSeq(self)
            else:
                return visitor.visitChildren(self)




    def attrSeq(self):

        localctx = RelayParser.AttrSeqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_attrSeq)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 387
            self.attr()
            self.state = 392
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==RelayParser.T__4:
                self.state = 388
                self.match(RelayParser.T__4)
                self.state = 389
                self.attr()
                self.state = 394
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AttrContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CNAME(self):
            return self.getToken(RelayParser.CNAME, 0)

        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_attr

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAttr" ):
                return visitor.visitAttr(self)
            else:
                return visitor.visitChildren(self)




    def attr(self):

        localctx = RelayParser.AttrContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_attr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 395
            self.match(RelayParser.CNAME)
            self.state = 396
            self.match(RelayParser.T__14)
            self.state = 397
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TypeExprContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_typeExpr

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class TypeParenContext(TypeExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.TypeExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def typeExpr(self):
            return self.getTypedRuleContext(RelayParser.TypeExprContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTypeParen" ):
                return visitor.visitTypeParen(self)
            else:
                return visitor.visitChildren(self)


    class TupleTypeContext(TypeExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.TypeExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def typeExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.TypeExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.TypeExprContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTupleType" ):
                return visitor.visitTupleType(self)
            else:
                return visitor.visitChildren(self)


    class TypeCallTypeContext(TypeExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.TypeExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def generalIdent(self):
            return self.getTypedRuleContext(RelayParser.GeneralIdentContext,0)

        def typeParamList(self):
            return self.getTypedRuleContext(RelayParser.TypeParamListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTypeCallType" ):
                return visitor.visitTypeCallType(self)
            else:
                return visitor.visitChildren(self)


    class TypeIdentTypeContext(TypeExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.TypeExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def generalIdent(self):
            return self.getTypedRuleContext(RelayParser.GeneralIdentContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTypeIdentType" ):
                return visitor.visitTypeIdentType(self)
            else:
                return visitor.visitChildren(self)


    class IncompleteTypeContext(TypeExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.TypeExprContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIncompleteType" ):
                return visitor.visitIncompleteType(self)
            else:
                return visitor.visitChildren(self)


    class TensorTypeContext(TypeExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.TypeExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def shapeList(self):
            return self.getTypedRuleContext(RelayParser.ShapeListContext,0)

        def typeExpr(self):
            return self.getTypedRuleContext(RelayParser.TypeExprContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTensorType" ):
                return visitor.visitTensorType(self)
            else:
                return visitor.visitChildren(self)


    class FuncTypeContext(TypeExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.TypeExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def typeExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.TypeExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.TypeExprContext,i)

        def typeParamList(self):
            return self.getTypedRuleContext(RelayParser.TypeParamListContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFuncType" ):
                return visitor.visitFuncType(self)
            else:
                return visitor.visitChildren(self)



    def typeExpr(self):

        localctx = RelayParser.TypeExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_typeExpr)
        self._la = 0 # Token type
        try:
            self.state = 450
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,47,self._ctx)
            if la_ == 1:
                localctx = RelayParser.TupleTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 399
                self.match(RelayParser.T__5)
                self.state = 400
                self.match(RelayParser.T__6)
                pass

            elif la_ == 2:
                localctx = RelayParser.TypeParenContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 401
                self.match(RelayParser.T__5)
                self.state = 402
                self.typeExpr()
                self.state = 403
                self.match(RelayParser.T__6)
                pass

            elif la_ == 3:
                localctx = RelayParser.TupleTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 405
                self.match(RelayParser.T__5)
                self.state = 406
                self.typeExpr()
                self.state = 407
                self.match(RelayParser.T__4)
                self.state = 408
                self.match(RelayParser.T__6)
                pass

            elif la_ == 4:
                localctx = RelayParser.TupleTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 410
                self.match(RelayParser.T__5)
                self.state = 411
                self.typeExpr()
                self.state = 414 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 412
                    self.match(RelayParser.T__4)
                    self.state = 413
                    self.typeExpr()
                    self.state = 416 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (_la==RelayParser.T__4):
                        break

                self.state = 418
                self.match(RelayParser.T__6)
                pass

            elif la_ == 5:
                localctx = RelayParser.TypeCallTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 420
                self.generalIdent()
                self.state = 421
                self.typeParamList()
                pass

            elif la_ == 6:
                localctx = RelayParser.TypeIdentTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 423
                self.generalIdent()
                pass

            elif la_ == 7:
                localctx = RelayParser.TensorTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 424
                self.match(RelayParser.T__26)
                self.state = 425
                self.match(RelayParser.T__7)
                self.state = 426
                self.shapeList()
                self.state = 427
                self.match(RelayParser.T__4)
                self.state = 428
                self.typeExpr()
                self.state = 429
                self.match(RelayParser.T__8)
                pass

            elif la_ == 8:
                localctx = RelayParser.FuncTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 8)
                self.state = 431
                self.match(RelayParser.T__17)
                self.state = 433
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__7:
                    self.state = 432
                    self.typeParamList()


                self.state = 435
                self.match(RelayParser.T__5)
                self.state = 444
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.T__3) | (1 << RelayParser.T__5) | (1 << RelayParser.T__17) | (1 << RelayParser.T__26) | (1 << RelayParser.CNAME))) != 0):
                    self.state = 436
                    self.typeExpr()
                    self.state = 441
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la==RelayParser.T__4:
                        self.state = 437
                        self.match(RelayParser.T__4)
                        self.state = 438
                        self.typeExpr()
                        self.state = 443
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)



                self.state = 446
                self.match(RelayParser.T__6)
                self.state = 447
                self.match(RelayParser.T__18)
                self.state = 448
                self.typeExpr()
                pass

            elif la_ == 9:
                localctx = RelayParser.IncompleteTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 9)
                self.state = 449
                self.match(RelayParser.T__3)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TypeParamListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def typeExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.TypeExprContext)
            else:
                return self.getTypedRuleContext(RelayParser.TypeExprContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_typeParamList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTypeParamList" ):
                return visitor.visitTypeParamList(self)
            else:
                return visitor.visitChildren(self)




    def typeParamList(self):

        localctx = RelayParser.TypeParamListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_typeParamList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 452
            self.match(RelayParser.T__7)
            self.state = 453
            self.typeExpr()
            self.state = 458
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==RelayParser.T__4:
                self.state = 454
                self.match(RelayParser.T__4)
                self.state = 455
                self.typeExpr()
                self.state = 460
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 461
            self.match(RelayParser.T__8)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ShapeListContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def shape(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.ShapeContext)
            else:
                return self.getTypedRuleContext(RelayParser.ShapeContext,i)


        def getRuleIndex(self):
            return RelayParser.RULE_shapeList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitShapeList" ):
                return visitor.visitShapeList(self)
            else:
                return visitor.visitChildren(self)




    def shapeList(self):

        localctx = RelayParser.ShapeListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_shapeList)
        self._la = 0 # Token type
        try:
            self.state = 476
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,50,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 463
                self.match(RelayParser.T__5)
                self.state = 464
                self.match(RelayParser.T__6)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 465
                self.match(RelayParser.T__5)
                self.state = 466
                self.shape()
                self.state = 469 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 467
                    self.match(RelayParser.T__4)
                    self.state = 468
                    self.shape()
                    self.state = 471 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (_la==RelayParser.T__4):
                        break

                self.state = 473
                self.match(RelayParser.T__6)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 475
                self.shape()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MetaContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CNAME(self):
            return self.getToken(RelayParser.CNAME, 0)

        def NAT(self):
            return self.getToken(RelayParser.NAT, 0)

        def getRuleIndex(self):
            return RelayParser.RULE_meta

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMeta" ):
                return visitor.visitMeta(self)
            else:
                return visitor.visitChildren(self)




    def meta(self):

        localctx = RelayParser.MetaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_meta)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 478
            self.match(RelayParser.T__27)
            self.state = 479
            self.match(RelayParser.T__7)
            self.state = 480
            self.match(RelayParser.CNAME)
            self.state = 481
            self.match(RelayParser.T__8)
            self.state = 482
            self.match(RelayParser.T__7)
            self.state = 483
            self.match(RelayParser.NAT)
            self.state = 484
            self.match(RelayParser.T__8)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ShapeContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_shape

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class ParensShapeContext(ShapeContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ShapeContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def shape(self):
            return self.getTypedRuleContext(RelayParser.ShapeContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitParensShape" ):
                return visitor.visitParensShape(self)
            else:
                return visitor.visitChildren(self)


    class MetaShapeContext(ShapeContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ShapeContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def meta(self):
            return self.getTypedRuleContext(RelayParser.MetaContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMetaShape" ):
                return visitor.visitMetaShape(self)
            else:
                return visitor.visitChildren(self)


    class IntShapeContext(ShapeContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ShapeContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NAT(self):
            return self.getToken(RelayParser.NAT, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIntShape" ):
                return visitor.visitIntShape(self)
            else:
                return visitor.visitChildren(self)



    def shape(self):

        localctx = RelayParser.ShapeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 60, self.RULE_shape)
        try:
            self.state = 492
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__27]:
                localctx = RelayParser.MetaShapeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 486
                self.meta()
                pass
            elif token in [RelayParser.T__5]:
                localctx = RelayParser.ParensShapeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 487
                self.match(RelayParser.T__5)
                self.state = 488
                self.shape()
                self.state = 489
                self.match(RelayParser.T__6)
                pass
            elif token in [RelayParser.NAT]:
                localctx = RelayParser.IntShapeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 491
                self.match(RelayParser.NAT)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BodyContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_body

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBody" ):
                return visitor.visitBody(self)
            else:
                return visitor.visitChildren(self)




    def body(self):

        localctx = RelayParser.BodyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 62, self.RULE_body)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 494
            self.match(RelayParser.T__11)
            self.state = 495
            self.expr(0)
            self.state = 496
            self.match(RelayParser.T__12)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ScalarContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RelayParser.RULE_scalar

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class ScalarFloatContext(ScalarContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ScalarContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def FLOAT(self):
            return self.getToken(RelayParser.FLOAT, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitScalarFloat" ):
                return visitor.visitScalarFloat(self)
            else:
                return visitor.visitChildren(self)


    class ScalarBoolContext(ScalarContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ScalarContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def BOOL_LIT(self):
            return self.getToken(RelayParser.BOOL_LIT, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitScalarBool" ):
                return visitor.visitScalarBool(self)
            else:
                return visitor.visitChildren(self)


    class ScalarIntContext(ScalarContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RelayParser.ScalarContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NAT(self):
            return self.getToken(RelayParser.NAT, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitScalarInt" ):
                return visitor.visitScalarInt(self)
            else:
                return visitor.visitChildren(self)



    def scalar(self):

        localctx = RelayParser.ScalarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 64, self.RULE_scalar)
        try:
            self.state = 501
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.FLOAT]:
                localctx = RelayParser.ScalarFloatContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 498
                self.match(RelayParser.FLOAT)
                pass
            elif token in [RelayParser.NAT]:
                localctx = RelayParser.ScalarIntContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 499
                self.match(RelayParser.NAT)
                pass
            elif token in [RelayParser.BOOL_LIT]:
                localctx = RelayParser.ScalarBoolContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 500
                self.match(RelayParser.BOOL_LIT)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IdentContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def generalIdent(self):
            return self.getTypedRuleContext(RelayParser.GeneralIdentContext,0)


        def globalVar(self):
            return self.getTypedRuleContext(RelayParser.GlobalVarContext,0)


        def localVar(self):
            return self.getTypedRuleContext(RelayParser.LocalVarContext,0)


        def graphVar(self):
            return self.getTypedRuleContext(RelayParser.GraphVarContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_ident

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIdent" ):
                return visitor.visitIdent(self)
            else:
                return visitor.visitChildren(self)




    def ident(self):

        localctx = RelayParser.IdentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 66, self.RULE_ident)
        try:
            self.state = 507
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,53,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 503
                self.generalIdent()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 504
                self.globalVar()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 505
                self.localVar()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 506
                self.graphVar()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[7] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 19)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 18)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 17)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 16)
         

            if predIndex == 4:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 5:
                return self.precpred(self._ctx, 21)
         

            if predIndex == 6:
                return self.precpred(self._ctx, 8)
         




