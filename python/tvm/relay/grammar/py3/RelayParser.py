# Generated from /Users/doobs/Code/repo/sampl/tvm/python/tvm/relay/grammar/Relay.g4 by ANTLR 4.7.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\61")
        buf.write("\u01f6\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7")
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
        buf.write("\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\5\t\u00b0\n\t\3\t\3\t")
        buf.write("\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3")
        buf.write("\t\3\t\3\t\3\t\5\t\u00c5\n\t\3\t\3\t\3\t\3\t\3\t\3\t\3")
        buf.write("\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t")
        buf.write("\3\t\3\t\3\t\7\t\u00de\n\t\f\t\16\t\u00e1\13\t\3\n\3\n")
        buf.write("\5\n\u00e5\n\n\3\n\3\n\3\n\3\n\3\n\5\n\u00ec\n\n\3\n\3")
        buf.write("\n\3\13\3\13\3\13\5\13\u00f3\n\13\3\13\3\13\3\13\3\13")
        buf.write("\3\13\5\13\u00fa\n\13\3\13\3\13\3\13\3\13\3\13\5\13\u0101")
        buf.write("\n\13\3\13\3\13\5\13\u0105\n\13\3\13\3\13\5\13\u0109\n")
        buf.write("\13\3\f\3\f\3\r\3\r\3\r\7\r\u0110\n\r\f\r\16\r\u0113\13")
        buf.write("\r\3\r\5\r\u0116\n\r\3\16\3\16\3\16\3\16\3\16\7\16\u011d")
        buf.write("\n\16\f\16\16\16\u0120\13\16\3\16\3\16\5\16\u0124\n\16")
        buf.write("\3\17\3\17\3\17\7\17\u0129\n\17\f\17\16\17\u012c\13\17")
        buf.write("\3\17\5\17\u012f\n\17\3\20\3\20\5\20\u0133\n\20\3\20\3")
        buf.write("\20\3\20\3\20\3\20\3\20\5\20\u013b\n\20\3\21\3\21\3\22")
        buf.write("\3\22\3\22\3\22\7\22\u0143\n\22\f\22\16\22\u0146\13\22")
        buf.write("\3\22\3\22\3\23\3\23\3\23\3\23\5\23\u014e\n\23\5\23\u0150")
        buf.write("\n\23\3\24\3\24\5\24\u0154\n\24\3\25\3\25\3\25\3\25\7")
        buf.write("\25\u015a\n\25\f\25\16\25\u015d\13\25\3\25\3\25\3\26\3")
        buf.write("\26\5\26\u0163\n\26\3\27\3\27\3\27\3\27\7\27\u0169\n\27")
        buf.write("\f\27\16\27\u016c\13\27\3\27\5\27\u016f\n\27\3\30\3\30")
        buf.write("\3\30\7\30\u0174\n\30\f\30\16\30\u0177\13\30\5\30\u0179")
        buf.write("\n\30\3\31\3\31\3\31\5\31\u017e\n\31\3\32\3\32\3\32\7")
        buf.write("\32\u0183\n\32\f\32\16\32\u0186\13\32\3\33\3\33\3\33\3")
        buf.write("\33\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34")
        buf.write("\3\34\6\34\u0197\n\34\r\34\16\34\u0198\3\34\3\34\3\34")
        buf.write("\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34\3\34")
        buf.write("\3\34\5\34\u01aa\n\34\3\34\3\34\3\34\3\34\7\34\u01b0\n")
        buf.write("\34\f\34\16\34\u01b3\13\34\5\34\u01b5\n\34\3\34\3\34\3")
        buf.write("\34\3\34\5\34\u01bb\n\34\3\35\3\35\3\35\3\35\7\35\u01c1")
        buf.write("\n\35\f\35\16\35\u01c4\13\35\3\35\3\35\3\36\3\36\3\36")
        buf.write("\3\36\3\36\3\36\6\36\u01ce\n\36\r\36\16\36\u01cf\3\36")
        buf.write("\3\36\3\36\5\36\u01d5\n\36\3\37\3\37\3\37\3\37\3\37\3")
        buf.write("\37\3\37\3\37\3 \3 \3 \3 \3 \3 \5 \u01e5\n \3!\3!\3!\3")
        buf.write("!\3\"\3\"\3\"\5\"\u01ee\n\"\3#\3#\3#\3#\5#\u01f4\n#\3")
        buf.write("#\2\3\20$\2\4\6\b\n\f\16\20\22\24\26\30\32\34\36 \"$&")
        buf.write("(*,.\60\62\64\668:<>@BD\2\b\4\2\6\6..\3\2#$\3\2%&\3\2")
        buf.write("\'*\3\2+,\3\2\31\32\2\u0225\2F\3\2\2\2\4U\3\2\2\2\6]\3")
        buf.write("\2\2\2\b`\3\2\2\2\nc\3\2\2\2\fn\3\2\2\2\16z\3\2\2\2\20")
        buf.write("\u00c4\3\2\2\2\22\u00e2\3\2\2\2\24\u0108\3\2\2\2\26\u010a")
        buf.write("\3\2\2\2\30\u010c\3\2\2\2\32\u0117\3\2\2\2\34\u0125\3")
        buf.write("\2\2\2\36\u0130\3\2\2\2 \u013c\3\2\2\2\"\u013e\3\2\2\2")
        buf.write("$\u014f\3\2\2\2&\u0151\3\2\2\2(\u0155\3\2\2\2*\u0162\3")
        buf.write("\2\2\2,\u016e\3\2\2\2.\u0178\3\2\2\2\60\u017a\3\2\2\2")
        buf.write("\62\u017f\3\2\2\2\64\u0187\3\2\2\2\66\u01ba\3\2\2\28\u01bc")
        buf.write("\3\2\2\2:\u01d4\3\2\2\2<\u01d6\3\2\2\2>\u01e4\3\2\2\2")
        buf.write("@\u01e6\3\2\2\2B\u01ed\3\2\2\2D\u01f3\3\2\2\2FN\7\36\2")
        buf.write("\2GI\5\24\13\2HG\3\2\2\2IL\3\2\2\2JH\3\2\2\2JK\3\2\2\2")
        buf.write("KO\3\2\2\2LJ\3\2\2\2MO\5\20\t\2NJ\3\2\2\2NM\3\2\2\2OQ")
        buf.write("\3\2\2\2PR\7\61\2\2QP\3\2\2\2QR\3\2\2\2RS\3\2\2\2ST\7")
        buf.write("\2\2\3T\3\3\2\2\2UZ\7.\2\2VW\7\3\2\2WY\7.\2\2XV\3\2\2")
        buf.write("\2Y\\\3\2\2\2ZX\3\2\2\2Z[\3\2\2\2[\5\3\2\2\2\\Z\3\2\2")
        buf.write("\2]^\7\4\2\2^_\7.\2\2_\7\3\2\2\2`a\7\5\2\2ab\t\2\2\2b")
        buf.write("\t\3\2\2\2cd\7\5\2\2de\7\60\2\2e\13\3\2\2\2fk\5\20\t\2")
        buf.write("gh\7\7\2\2hj\5\20\t\2ig\3\2\2\2jm\3\2\2\2ki\3\2\2\2kl")
        buf.write("\3\2\2\2lo\3\2\2\2mk\3\2\2\2nf\3\2\2\2no\3\2\2\2o\r\3")
        buf.write("\2\2\2p{\5\f\7\2qr\5\20\t\2rs\7\7\2\2su\3\2\2\2tq\3\2")
        buf.write("\2\2ux\3\2\2\2vt\3\2\2\2vw\3\2\2\2wy\3\2\2\2xv\3\2\2\2")
        buf.write("y{\5\62\32\2zp\3\2\2\2zv\3\2\2\2{\17\3\2\2\2|}\b\t\1\2")
        buf.write("}~\7\b\2\2~\177\5\20\t\2\177\u0080\7\t\2\2\u0080\u00c5")
        buf.write("\3\2\2\2\u0081\u0082\7&\2\2\u0082\u00c5\5\20\t\26\u0083")
        buf.write("\u00c5\5\22\n\2\u0084\u0085\7\b\2\2\u0085\u00c5\7\t\2")
        buf.write("\2\u0086\u0087\7\b\2\2\u0087\u0088\5\20\t\2\u0088\u0089")
        buf.write("\7\7\2\2\u0089\u008a\7\t\2\2\u008a\u00c5\3\2\2\2\u008b")
        buf.write("\u008c\7\b\2\2\u008c\u008f\5\20\t\2\u008d\u008e\7\7\2")
        buf.write("\2\u008e\u0090\5\20\t\2\u008f\u008d\3\2\2\2\u0090\u0091")
        buf.write("\3\2\2\2\u0091\u008f\3\2\2\2\u0091\u0092\3\2\2\2\u0092")
        buf.write("\u0093\3\2\2\2\u0093\u0094\7\t\2\2\u0094\u00c5\3\2\2\2")
        buf.write("\u0095\u009e\7\n\2\2\u0096\u009b\5\20\t\2\u0097\u0098")
        buf.write("\7\7\2\2\u0098\u009a\5\20\t\2\u0099\u0097\3\2\2\2\u009a")
        buf.write("\u009d\3\2\2\2\u009b\u0099\3\2\2\2\u009b\u009c\3\2\2\2")
        buf.write("\u009c\u009f\3\2\2\2\u009d\u009b\3\2\2\2\u009e\u0096\3")
        buf.write("\2\2\2\u009e\u009f\3\2\2\2\u009f\u00a0\3\2\2\2\u00a0\u00c5")
        buf.write("\7\13\2\2\u00a1\u00a2\7\f\2\2\u00a2\u00a3\7\b\2\2\u00a3")
        buf.write("\u00a4\5\20\t\2\u00a4\u00a5\7\t\2\2\u00a5\u00a6\5@!\2")
        buf.write("\u00a6\u00a7\7\r\2\2\u00a7\u00a8\5@!\2\u00a8\u00c5\3\2")
        buf.write("\2\2\u00a9\u00aa\5 \21\2\u00aa\u00ab\7\b\2\2\u00ab\u00ac")
        buf.write("\5\20\t\2\u00ac\u00ad\7\t\2\2\u00ad\u00af\7\16\2\2\u00ae")
        buf.write("\u00b0\5\34\17\2\u00af\u00ae\3\2\2\2\u00af\u00b0\3\2\2")
        buf.write("\2\u00b0\u00b1\3\2\2\2\u00b1\u00b2\7\17\2\2\u00b2\u00c5")
        buf.write("\3\2\2\2\u00b3\u00b4\7\20\2\2\u00b4\u00b5\5\60\31\2\u00b5")
        buf.write("\u00b6\7\21\2\2\u00b6\u00b7\5\20\t\2\u00b7\u00b8\7\22")
        buf.write("\2\2\u00b8\u00b9\5\20\t\t\u00b9\u00c5\3\2\2\2\u00ba\u00bb")
        buf.write("\5\n\6\2\u00bb\u00bc\7\21\2\2\u00bc\u00bd\5\20\t\2\u00bd")
        buf.write("\u00be\7\22\2\2\u00be\u00bf\5\20\t\7\u00bf\u00c5\3\2\2")
        buf.write("\2\u00c0\u00c5\5D#\2\u00c1\u00c5\5B\"\2\u00c2\u00c5\5")
        buf.write("<\37\2\u00c3\u00c5\7\"\2\2\u00c4|\3\2\2\2\u00c4\u0081")
        buf.write("\3\2\2\2\u00c4\u0083\3\2\2\2\u00c4\u0084\3\2\2\2\u00c4")
        buf.write("\u0086\3\2\2\2\u00c4\u008b\3\2\2\2\u00c4\u0095\3\2\2\2")
        buf.write("\u00c4\u00a1\3\2\2\2\u00c4\u00a9\3\2\2\2\u00c4\u00b3\3")
        buf.write("\2\2\2\u00c4\u00ba\3\2\2\2\u00c4\u00c0\3\2\2\2\u00c4\u00c1")
        buf.write("\3\2\2\2\u00c4\u00c2\3\2\2\2\u00c4\u00c3\3\2\2\2\u00c5")
        buf.write("\u00df\3\2\2\2\u00c6\u00c7\f\25\2\2\u00c7\u00c8\t\3\2")
        buf.write("\2\u00c8\u00de\5\20\t\26\u00c9\u00ca\f\24\2\2\u00ca\u00cb")
        buf.write("\t\4\2\2\u00cb\u00de\5\20\t\25\u00cc\u00cd\f\23\2\2\u00cd")
        buf.write("\u00ce\t\5\2\2\u00ce\u00de\5\20\t\24\u00cf\u00d0\f\22")
        buf.write("\2\2\u00d0\u00d1\t\6\2\2\u00d1\u00de\5\20\t\23\u00d2\u00d3")
        buf.write("\f\b\2\2\u00d3\u00d4\7\23\2\2\u00d4\u00de\5\20\t\t\u00d5")
        buf.write("\u00d6\f\27\2\2\u00d6\u00d7\7\b\2\2\u00d7\u00d8\5\16\b")
        buf.write("\2\u00d8\u00d9\7\t\2\2\u00d9\u00de\3\2\2\2\u00da\u00db")
        buf.write("\f\n\2\2\u00db\u00dc\7\3\2\2\u00dc\u00de\7\60\2\2\u00dd")
        buf.write("\u00c6\3\2\2\2\u00dd\u00c9\3\2\2\2\u00dd\u00cc\3\2\2\2")
        buf.write("\u00dd\u00cf\3\2\2\2\u00dd\u00d2\3\2\2\2\u00dd\u00d5\3")
        buf.write("\2\2\2\u00dd\u00da\3\2\2\2\u00de\u00e1\3\2\2\2\u00df\u00dd")
        buf.write("\3\2\2\2\u00df\u00e0\3\2\2\2\u00e0\21\3\2\2\2\u00e1\u00df")
        buf.write("\3\2\2\2\u00e2\u00e4\7\24\2\2\u00e3\u00e5\58\35\2\u00e4")
        buf.write("\u00e3\3\2\2\2\u00e4\u00e5\3\2\2\2\u00e5\u00e6\3\2\2\2")
        buf.write("\u00e6\u00e7\7\b\2\2\u00e7\u00e8\5,\27\2\u00e8\u00eb\7")
        buf.write("\t\2\2\u00e9\u00ea\7\25\2\2\u00ea\u00ec\5\66\34\2\u00eb")
        buf.write("\u00e9\3\2\2\2\u00eb\u00ec\3\2\2\2\u00ec\u00ed\3\2\2\2")
        buf.write("\u00ed\u00ee\5@!\2\u00ee\23\3\2\2\2\u00ef\u00f0\7\26\2")
        buf.write("\2\u00f0\u00f2\5\6\4\2\u00f1\u00f3\58\35\2\u00f2\u00f1")
        buf.write("\3\2\2\2\u00f2\u00f3\3\2\2\2\u00f3\u00f4\3\2\2\2\u00f4")
        buf.write("\u00f5\7\b\2\2\u00f5\u00f6\5,\27\2\u00f6\u00f9\7\t\2\2")
        buf.write("\u00f7\u00f8\7\25\2\2\u00f8\u00fa\5\66\34\2\u00f9\u00f7")
        buf.write("\3\2\2\2\u00f9\u00fa\3\2\2\2\u00fa\u00fb\3\2\2\2\u00fb")
        buf.write("\u00fc\5@!\2\u00fc\u0109\3\2\2\2\u00fd\u00fe\7\27\2\2")
        buf.write("\u00fe\u0100\5\4\3\2\u00ff\u0101\58\35\2\u0100\u00ff\3")
        buf.write("\2\2\2\u0100\u0101\3\2\2\2\u0101\u0102\3\2\2\2\u0102\u0104")
        buf.write("\7\16\2\2\u0103\u0105\5\30\r\2\u0104\u0103\3\2\2\2\u0104")
        buf.write("\u0105\3\2\2\2\u0105\u0106\3\2\2\2\u0106\u0107\7\17\2")
        buf.write("\2\u0107\u0109\3\2\2\2\u0108\u00ef\3\2\2\2\u0108\u00fd")
        buf.write("\3\2\2\2\u0109\25\3\2\2\2\u010a\u010b\7.\2\2\u010b\27")
        buf.write("\3\2\2\2\u010c\u0111\5\32\16\2\u010d\u010e\7\7\2\2\u010e")
        buf.write("\u0110\5\32\16\2\u010f\u010d\3\2\2\2\u0110\u0113\3\2\2")
        buf.write("\2\u0111\u010f\3\2\2\2\u0111\u0112\3\2\2\2\u0112\u0115")
        buf.write("\3\2\2\2\u0113\u0111\3\2\2\2\u0114\u0116\7\7\2\2\u0115")
        buf.write("\u0114\3\2\2\2\u0115\u0116\3\2\2\2\u0116\31\3\2\2\2\u0117")
        buf.write("\u0123\5\26\f\2\u0118\u0119\7\b\2\2\u0119\u011e\5\66\34")
        buf.write("\2\u011a\u011b\7\7\2\2\u011b\u011d\5\66\34\2\u011c\u011a")
        buf.write("\3\2\2\2\u011d\u0120\3\2\2\2\u011e\u011c\3\2\2\2\u011e")
        buf.write("\u011f\3\2\2\2\u011f\u0121\3\2\2\2\u0120\u011e\3\2\2\2")
        buf.write("\u0121\u0122\7\t\2\2\u0122\u0124\3\2\2\2\u0123\u0118\3")
        buf.write("\2\2\2\u0123\u0124\3\2\2\2\u0124\33\3\2\2\2\u0125\u012a")
        buf.write("\5\36\20\2\u0126\u0127\7\7\2\2\u0127\u0129\5\36\20\2\u0128")
        buf.write("\u0126\3\2\2\2\u0129\u012c\3\2\2\2\u012a\u0128\3\2\2\2")
        buf.write("\u012a\u012b\3\2\2\2\u012b\u012e\3\2\2\2\u012c\u012a\3")
        buf.write("\2\2\2\u012d\u012f\7\7\2\2\u012e\u012d\3\2\2\2\u012e\u012f")
        buf.write("\3\2\2\2\u012f\35\3\2\2\2\u0130\u0132\5\26\f\2\u0131\u0133")
        buf.write("\5\"\22\2\u0132\u0131\3\2\2\2\u0132\u0133\3\2\2\2\u0133")
        buf.write("\u0134\3\2\2\2\u0134\u013a\7\30\2\2\u0135\u0136\7\16\2")
        buf.write("\2\u0136\u0137\5\20\t\2\u0137\u0138\7\17\2\2\u0138\u013b")
        buf.write("\3\2\2\2\u0139\u013b\5\20\t\2\u013a\u0135\3\2\2\2\u013a")
        buf.write("\u0139\3\2\2\2\u013b\37\3\2\2\2\u013c\u013d\t\7\2\2\u013d")
        buf.write("!\3\2\2\2\u013e\u013f\7\b\2\2\u013f\u0144\5$\23\2\u0140")
        buf.write("\u0141\7\7\2\2\u0141\u0143\5$\23\2\u0142\u0140\3\2\2\2")
        buf.write("\u0143\u0146\3\2\2\2\u0144\u0142\3\2\2\2\u0144\u0145\3")
        buf.write("\2\2\2\u0145\u0147\3\2\2\2\u0146\u0144\3\2\2\2\u0147\u0148")
        buf.write("\7\t\2\2\u0148#\3\2\2\2\u0149\u0150\7\6\2\2\u014a\u014d")
        buf.write("\5\b\5\2\u014b\u014c\7\33\2\2\u014c\u014e\5\66\34\2\u014d")
        buf.write("\u014b\3\2\2\2\u014d\u014e\3\2\2\2\u014e\u0150\3\2\2\2")
        buf.write("\u014f\u0149\3\2\2\2\u014f\u014a\3\2\2\2\u0150%\3\2\2")
        buf.write("\2\u0151\u0153\5\26\f\2\u0152\u0154\5(\25\2\u0153\u0152")
        buf.write("\3\2\2\2\u0153\u0154\3\2\2\2\u0154\'\3\2\2\2\u0155\u0156")
        buf.write("\7\b\2\2\u0156\u015b\5*\26\2\u0157\u0158\7\7\2\2\u0158")
        buf.write("\u015a\5*\26\2\u0159\u0157\3\2\2\2\u015a\u015d\3\2\2\2")
        buf.write("\u015b\u0159\3\2\2\2\u015b\u015c\3\2\2\2\u015c\u015e\3")
        buf.write("\2\2\2\u015d\u015b\3\2\2\2\u015e\u015f\7\t\2\2\u015f)")
        buf.write("\3\2\2\2\u0160\u0163\5\b\5\2\u0161\u0163\5\26\f\2\u0162")
        buf.write("\u0160\3\2\2\2\u0162\u0161\3\2\2\2\u0163+\3\2\2\2\u0164")
        buf.write("\u016f\5.\30\2\u0165\u0166\5\60\31\2\u0166\u0167\7\7\2")
        buf.write("\2\u0167\u0169\3\2\2\2\u0168\u0165\3\2\2\2\u0169\u016c")
        buf.write("\3\2\2\2\u016a\u0168\3\2\2\2\u016a\u016b\3\2\2\2\u016b")
        buf.write("\u016d\3\2\2\2\u016c\u016a\3\2\2\2\u016d\u016f\5\62\32")
        buf.write("\2\u016e\u0164\3\2\2\2\u016e\u016a\3\2\2\2\u016f-\3\2")
        buf.write("\2\2\u0170\u0175\5\60\31\2\u0171\u0172\7\7\2\2\u0172\u0174")
        buf.write("\5\60\31\2\u0173\u0171\3\2\2\2\u0174\u0177\3\2\2\2\u0175")
        buf.write("\u0173\3\2\2\2\u0175\u0176\3\2\2\2\u0176\u0179\3\2\2\2")
        buf.write("\u0177\u0175\3\2\2\2\u0178\u0170\3\2\2\2\u0178\u0179\3")
        buf.write("\2\2\2\u0179/\3\2\2\2\u017a\u017d\5\b\5\2\u017b\u017c")
        buf.write("\7\33\2\2\u017c\u017e\5\66\34\2\u017d\u017b\3\2\2\2\u017d")
        buf.write("\u017e\3\2\2\2\u017e\61\3\2\2\2\u017f\u0184\5\64\33\2")
        buf.write("\u0180\u0181\7\7\2\2\u0181\u0183\5\64\33\2\u0182\u0180")
        buf.write("\3\2\2\2\u0183\u0186\3\2\2\2\u0184\u0182\3\2\2\2\u0184")
        buf.write("\u0185\3\2\2\2\u0185\63\3\2\2\2\u0186\u0184\3\2\2\2\u0187")
        buf.write("\u0188\7.\2\2\u0188\u0189\7\21\2\2\u0189\u018a\5\20\t")
        buf.write("\2\u018a\65\3\2\2\2\u018b\u018c\7\b\2\2\u018c\u01bb\7")
        buf.write("\t\2\2\u018d\u018e\7\b\2\2\u018e\u018f\5\66\34\2\u018f")
        buf.write("\u0190\7\7\2\2\u0190\u0191\7\t\2\2\u0191\u01bb\3\2\2\2")
        buf.write("\u0192\u0193\7\b\2\2\u0193\u0196\5\66\34\2\u0194\u0195")
        buf.write("\7\7\2\2\u0195\u0197\5\66\34\2\u0196\u0194\3\2\2\2\u0197")
        buf.write("\u0198\3\2\2\2\u0198\u0196\3\2\2\2\u0198\u0199\3\2\2\2")
        buf.write("\u0199\u019a\3\2\2\2\u019a\u019b\7\t\2\2\u019b\u01bb\3")
        buf.write("\2\2\2\u019c\u019d\5\4\3\2\u019d\u019e\58\35\2\u019e\u01bb")
        buf.write("\3\2\2\2\u019f\u01bb\5\4\3\2\u01a0\u01a1\7\34\2\2\u01a1")
        buf.write("\u01a2\7\n\2\2\u01a2\u01a3\5:\36\2\u01a3\u01a4\7\7\2\2")
        buf.write("\u01a4\u01a5\5\66\34\2\u01a5\u01a6\7\13\2\2\u01a6\u01bb")
        buf.write("\3\2\2\2\u01a7\u01a9\7\24\2\2\u01a8\u01aa\58\35\2\u01a9")
        buf.write("\u01a8\3\2\2\2\u01a9\u01aa\3\2\2\2\u01aa\u01ab\3\2\2\2")
        buf.write("\u01ab\u01b4\7\b\2\2\u01ac\u01b1\5\66\34\2\u01ad\u01ae")
        buf.write("\7\7\2\2\u01ae\u01b0\5\66\34\2\u01af\u01ad\3\2\2\2\u01b0")
        buf.write("\u01b3\3\2\2\2\u01b1\u01af\3\2\2\2\u01b1\u01b2\3\2\2\2")
        buf.write("\u01b2\u01b5\3\2\2\2\u01b3\u01b1\3\2\2\2\u01b4\u01ac\3")
        buf.write("\2\2\2\u01b4\u01b5\3\2\2\2\u01b5\u01b6\3\2\2\2\u01b6\u01b7")
        buf.write("\7\t\2\2\u01b7\u01b8\7\25\2\2\u01b8\u01bb\5\66\34\2\u01b9")
        buf.write("\u01bb\7\6\2\2\u01ba\u018b\3\2\2\2\u01ba\u018d\3\2\2\2")
        buf.write("\u01ba\u0192\3\2\2\2\u01ba\u019c\3\2\2\2\u01ba\u019f\3")
        buf.write("\2\2\2\u01ba\u01a0\3\2\2\2\u01ba\u01a7\3\2\2\2\u01ba\u01b9")
        buf.write("\3\2\2\2\u01bb\67\3\2\2\2\u01bc\u01bd\7\n\2\2\u01bd\u01c2")
        buf.write("\5\4\3\2\u01be\u01bf\7\7\2\2\u01bf\u01c1\5\4\3\2\u01c0")
        buf.write("\u01be\3\2\2\2\u01c1\u01c4\3\2\2\2\u01c2\u01c0\3\2\2\2")
        buf.write("\u01c2\u01c3\3\2\2\2\u01c3\u01c5\3\2\2\2\u01c4\u01c2\3")
        buf.write("\2\2\2\u01c5\u01c6\7\13\2\2\u01c69\3\2\2\2\u01c7\u01c8")
        buf.write("\7\b\2\2\u01c8\u01d5\7\t\2\2\u01c9\u01ca\7\b\2\2\u01ca")
        buf.write("\u01cd\5> \2\u01cb\u01cc\7\7\2\2\u01cc\u01ce\5> \2\u01cd")
        buf.write("\u01cb\3\2\2\2\u01ce\u01cf\3\2\2\2\u01cf\u01cd\3\2\2\2")
        buf.write("\u01cf\u01d0\3\2\2\2\u01d0\u01d1\3\2\2\2\u01d1\u01d2\7")
        buf.write("\t\2\2\u01d2\u01d5\3\2\2\2\u01d3\u01d5\5> \2\u01d4\u01c7")
        buf.write("\3\2\2\2\u01d4\u01c9\3\2\2\2\u01d4\u01d3\3\2\2\2\u01d5")
        buf.write(";\3\2\2\2\u01d6\u01d7\7\35\2\2\u01d7\u01d8\7\n\2\2\u01d8")
        buf.write("\u01d9\7.\2\2\u01d9\u01da\7\13\2\2\u01da\u01db\7\n\2\2")
        buf.write("\u01db\u01dc\7\60\2\2\u01dc\u01dd\7\13\2\2\u01dd=\3\2")
        buf.write("\2\2\u01de\u01e5\5<\37\2\u01df\u01e0\7\b\2\2\u01e0\u01e1")
        buf.write("\5> \2\u01e1\u01e2\7\t\2\2\u01e2\u01e5\3\2\2\2\u01e3\u01e5")
        buf.write("\7\60\2\2\u01e4\u01de\3\2\2\2\u01e4\u01df\3\2\2\2\u01e4")
        buf.write("\u01e3\3\2\2\2\u01e5?\3\2\2\2\u01e6\u01e7\7\16\2\2\u01e7")
        buf.write("\u01e8\5\20\t\2\u01e8\u01e9\7\17\2\2\u01e9A\3\2\2\2\u01ea")
        buf.write("\u01ee\7/\2\2\u01eb\u01ee\7\60\2\2\u01ec\u01ee\7-\2\2")
        buf.write("\u01ed\u01ea\3\2\2\2\u01ed\u01eb\3\2\2\2\u01ed\u01ec\3")
        buf.write("\2\2\2\u01eeC\3\2\2\2\u01ef\u01f4\5\4\3\2\u01f0\u01f4")
        buf.write("\5\6\4\2\u01f1\u01f4\5\b\5\2\u01f2\u01f4\5\n\6\2\u01f3")
        buf.write("\u01ef\3\2\2\2\u01f3\u01f0\3\2\2\2\u01f3\u01f1\3\2\2\2")
        buf.write("\u01f3\u01f2\3\2\2\2\u01f4E\3\2\2\2\67JNQZknvz\u0091\u009b")
        buf.write("\u009e\u00af\u00c4\u00dd\u00df\u00e4\u00eb\u00f2\u00f9")
        buf.write("\u0100\u0104\u0108\u0111\u0115\u011e\u0123\u012a\u012e")
        buf.write("\u0132\u013a\u0144\u014d\u014f\u0153\u015b\u0162\u016a")
        buf.write("\u016e\u0175\u0178\u017d\u0184\u0198\u01a9\u01b1\u01b4")
        buf.write("\u01ba\u01c2\u01cf\u01d4\u01e4\u01ed\u01f3")
        return buf.getvalue()


class RelayParser ( Parser ):

    grammarFileName = "Relay.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'.'", "'@'", "'%'", "'_'", "','", "'('", 
                     "')'", "'['", "']'", "'if'", "'else'", "'{'", "'}'", 
                     "'let'", "'='", "';'", "';;'", "'fn'", "'->'", "'def'", 
                     "'type'", "'=>'", "'match'", "'match?'", "':'", "'Tensor'", 
                     "'meta'", "'v0.0.4'", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'*'", "'/'", "'+'", "'-'", "'<'", "'>'", 
                     "'<='", "'>='", "'=='", "'!='" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "SEMVER", "COMMENT", "WS", "LINE_COMMENT", "QUOTED_STRING", 
                      "MUL", "DIV", "ADD", "SUB", "LT", "GT", "LE", "GE", 
                      "EQ", "NE", "BOOL_LIT", "CNAME", "FLOAT", "NAT", "METADATA" ]

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
    SEMVER=28
    COMMENT=29
    WS=30
    LINE_COMMENT=31
    QUOTED_STRING=32
    MUL=33
    DIV=34
    ADD=35
    SUB=36
    LT=37
    GT=38
    LE=39
    GE=40
    EQ=41
    NE=42
    BOOL_LIT=43
    CNAME=44
    FLOAT=45
    NAT=46
    METADATA=47

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
            if token in [RelayParser.EOF, RelayParser.T__19, RelayParser.T__20, RelayParser.METADATA]:
                self.state = 72
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==RelayParser.T__19 or _la==RelayParser.T__20:
                    self.state = 69
                    self.defn()
                    self.state = 74
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                pass
            elif token in [RelayParser.T__1, RelayParser.T__2, RelayParser.T__5, RelayParser.T__7, RelayParser.T__9, RelayParser.T__13, RelayParser.T__17, RelayParser.T__22, RelayParser.T__23, RelayParser.T__26, RelayParser.QUOTED_STRING, RelayParser.SUB, RelayParser.BOOL_LIT, RelayParser.CNAME, RelayParser.FLOAT, RelayParser.NAT]:
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
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.T__1) | (1 << RelayParser.T__2) | (1 << RelayParser.T__5) | (1 << RelayParser.T__7) | (1 << RelayParser.T__9) | (1 << RelayParser.T__13) | (1 << RelayParser.T__17) | (1 << RelayParser.T__22) | (1 << RelayParser.T__23) | (1 << RelayParser.T__26) | (1 << RelayParser.QUOTED_STRING) | (1 << RelayParser.SUB) | (1 << RelayParser.BOOL_LIT) | (1 << RelayParser.CNAME) | (1 << RelayParser.FLOAT) | (1 << RelayParser.NAT))) != 0):
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
            self.state = 194
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
                if (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.T__1) | (1 << RelayParser.T__2) | (1 << RelayParser.T__5) | (1 << RelayParser.T__7) | (1 << RelayParser.T__9) | (1 << RelayParser.T__13) | (1 << RelayParser.T__17) | (1 << RelayParser.T__22) | (1 << RelayParser.T__23) | (1 << RelayParser.T__26) | (1 << RelayParser.QUOTED_STRING) | (1 << RelayParser.SUB) | (1 << RelayParser.BOOL_LIT) | (1 << RelayParser.CNAME) | (1 << RelayParser.FLOAT) | (1 << RelayParser.NAT))) != 0):
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
                self.match(RelayParser.T__5)
                self.state = 169
                self.expr(0)
                self.state = 170
                self.match(RelayParser.T__6)
                self.state = 171
                self.match(RelayParser.T__11)
                self.state = 173
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.CNAME:
                    self.state = 172
                    self.matchClauseList()


                self.state = 175
                self.match(RelayParser.T__12)
                pass

            elif la_ == 10:
                localctx = RelayParser.LetContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 177
                self.match(RelayParser.T__13)
                self.state = 178
                self.var()
                self.state = 179
                self.match(RelayParser.T__14)
                self.state = 180
                self.expr(0)
                self.state = 181
                self.match(RelayParser.T__15)
                self.state = 182
                self.expr(7)
                pass

            elif la_ == 11:
                localctx = RelayParser.GraphContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 184
                self.graphVar()
                self.state = 185
                self.match(RelayParser.T__14)
                self.state = 186
                self.expr(0)
                self.state = 187
                self.match(RelayParser.T__15)
                self.state = 188
                self.expr(5)
                pass

            elif la_ == 12:
                localctx = RelayParser.IdentExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 190
                self.ident()
                pass

            elif la_ == 13:
                localctx = RelayParser.ScalarExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 191
                self.scalar()
                pass

            elif la_ == 14:
                localctx = RelayParser.MetaExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 192
                self.meta()
                pass

            elif la_ == 15:
                localctx = RelayParser.StringExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 193
                self.match(RelayParser.QUOTED_STRING)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 221
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,14,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 219
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
                    if la_ == 1:
                        localctx = RelayParser.BinOpContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 196
                        if not self.precpred(self._ctx, 19):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 19)")
                        self.state = 197
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not(_la==RelayParser.MUL or _la==RelayParser.DIV):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 198
                        self.expr(20)
                        pass

                    elif la_ == 2:
                        localctx = RelayParser.BinOpContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 199
                        if not self.precpred(self._ctx, 18):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 18)")
                        self.state = 200
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not(_la==RelayParser.ADD or _la==RelayParser.SUB):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 201
                        self.expr(19)
                        pass

                    elif la_ == 3:
                        localctx = RelayParser.BinOpContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 202
                        if not self.precpred(self._ctx, 17):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 17)")
                        self.state = 203
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.LT) | (1 << RelayParser.GT) | (1 << RelayParser.LE) | (1 << RelayParser.GE))) != 0)):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 204
                        self.expr(18)
                        pass

                    elif la_ == 4:
                        localctx = RelayParser.BinOpContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 205
                        if not self.precpred(self._ctx, 16):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 16)")
                        self.state = 206
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not(_la==RelayParser.EQ or _la==RelayParser.NE):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 207
                        self.expr(17)
                        pass

                    elif la_ == 5:
                        localctx = RelayParser.LetContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 208
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 209
                        self.match(RelayParser.T__16)
                        self.state = 210
                        self.expr(7)
                        pass

                    elif la_ == 6:
                        localctx = RelayParser.CallContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 211
                        if not self.precpred(self._ctx, 21):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 21)")
                        self.state = 212
                        self.match(RelayParser.T__5)
                        self.state = 213
                        self.callList()
                        self.state = 214
                        self.match(RelayParser.T__6)
                        pass

                    elif la_ == 7:
                        localctx = RelayParser.ProjectionContext(self, RelayParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 216
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 217
                        self.match(RelayParser.T__0)
                        self.state = 218
                        self.match(RelayParser.NAT)
                        pass

             
                self.state = 223
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
            self.state = 224
            self.match(RelayParser.T__17)
            self.state = 226
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__7:
                self.state = 225
                self.typeParamList()


            self.state = 228
            self.match(RelayParser.T__5)
            self.state = 229
            self.argList()
            self.state = 230
            self.match(RelayParser.T__6)
            self.state = 233
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__18:
                self.state = 231
                self.match(RelayParser.T__18)
                self.state = 232
                self.typeExpr()


            self.state = 235
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
            self.state = 262
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__19]:
                localctx = RelayParser.FuncDefnContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 237
                self.match(RelayParser.T__19)
                self.state = 238
                self.globalVar()
                self.state = 240
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__7:
                    self.state = 239
                    self.typeParamList()


                self.state = 242
                self.match(RelayParser.T__5)
                self.state = 243
                self.argList()
                self.state = 244
                self.match(RelayParser.T__6)
                self.state = 247
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__18:
                    self.state = 245
                    self.match(RelayParser.T__18)
                    self.state = 246
                    self.typeExpr()


                self.state = 249
                self.body()
                pass
            elif token in [RelayParser.T__20]:
                localctx = RelayParser.AdtDefnContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 251
                self.match(RelayParser.T__20)
                self.state = 252
                self.generalIdent()
                self.state = 254
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__7:
                    self.state = 253
                    self.typeParamList()


                self.state = 256
                self.match(RelayParser.T__11)
                self.state = 258
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.CNAME:
                    self.state = 257
                    self.adtConsDefnList()


                self.state = 260
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
            self.state = 264
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
            self.state = 266
            self.adtConsDefn()
            self.state = 271
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,22,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 267
                    self.match(RelayParser.T__4)
                    self.state = 268
                    self.adtConsDefn() 
                self.state = 273
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,22,self._ctx)

            self.state = 275
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__4:
                self.state = 274
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
            self.state = 277
            self.constructorName()
            self.state = 289
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__5:
                self.state = 278
                self.match(RelayParser.T__5)
                self.state = 279
                self.typeExpr()
                self.state = 284
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==RelayParser.T__4:
                    self.state = 280
                    self.match(RelayParser.T__4)
                    self.state = 281
                    self.typeExpr()
                    self.state = 286
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 287
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
            self.state = 291
            self.matchClause()
            self.state = 296
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,26,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 292
                    self.match(RelayParser.T__4)
                    self.state = 293
                    self.matchClause() 
                self.state = 298
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,26,self._ctx)

            self.state = 300
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__4:
                self.state = 299
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

        def constructorName(self):
            return self.getTypedRuleContext(RelayParser.ConstructorNameContext,0)


        def expr(self):
            return self.getTypedRuleContext(RelayParser.ExprContext,0)


        def patternList(self):
            return self.getTypedRuleContext(RelayParser.PatternListContext,0)


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
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 302
            self.constructorName()
            self.state = 304
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__5:
                self.state = 303
                self.patternList()


            self.state = 306
            self.match(RelayParser.T__21)
            self.state = 312
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__11]:
                self.state = 307
                self.match(RelayParser.T__11)
                self.state = 308
                self.expr(0)
                self.state = 309
                self.match(RelayParser.T__12)
                pass
            elif token in [RelayParser.T__1, RelayParser.T__2, RelayParser.T__5, RelayParser.T__7, RelayParser.T__9, RelayParser.T__13, RelayParser.T__17, RelayParser.T__22, RelayParser.T__23, RelayParser.T__26, RelayParser.QUOTED_STRING, RelayParser.SUB, RelayParser.BOOL_LIT, RelayParser.CNAME, RelayParser.FLOAT, RelayParser.NAT]:
                self.state = 311
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
            self.state = 314
            _la = self._input.LA(1)
            if not(_la==RelayParser.T__22 or _la==RelayParser.T__23):
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
            self.state = 316
            self.match(RelayParser.T__5)
            self.state = 317
            self.pattern()
            self.state = 322
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==RelayParser.T__4:
                self.state = 318
                self.match(RelayParser.T__4)
                self.state = 319
                self.pattern()
                self.state = 324
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 325
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

        def localVar(self):
            return self.getTypedRuleContext(RelayParser.LocalVarContext,0)


        def typeExpr(self):
            return self.getTypedRuleContext(RelayParser.TypeExprContext,0)


        def getRuleIndex(self):
            return RelayParser.RULE_pattern

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPattern" ):
                return visitor.visitPattern(self)
            else:
                return visitor.visitChildren(self)




    def pattern(self):

        localctx = RelayParser.PatternContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_pattern)
        self._la = 0 # Token type
        try:
            self.state = 333
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__3]:
                self.enterOuterAlt(localctx, 1)
                self.state = 327
                self.match(RelayParser.T__3)
                pass
            elif token in [RelayParser.T__2]:
                self.enterOuterAlt(localctx, 2)
                self.state = 328
                self.localVar()
                self.state = 331
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__24:
                    self.state = 329
                    self.match(RelayParser.T__24)
                    self.state = 330
                    self.typeExpr()


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
            self.state = 335
            self.constructorName()
            self.state = 337
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__5:
                self.state = 336
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
            self.state = 339
            self.match(RelayParser.T__5)
            self.state = 340
            self.adtConsParam()
            self.state = 345
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==RelayParser.T__4:
                self.state = 341
                self.match(RelayParser.T__4)
                self.state = 342
                self.adtConsParam()
                self.state = 347
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 348
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
            self.state = 352
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__2]:
                self.enterOuterAlt(localctx, 1)
                self.state = 350
                self.localVar()
                pass
            elif token in [RelayParser.CNAME]:
                self.enterOuterAlt(localctx, 2)
                self.state = 351
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
            self.state = 364
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,37,self._ctx)
            if la_ == 1:
                localctx = RelayParser.ArgNoAttrContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 354
                self.varList()
                pass

            elif la_ == 2:
                localctx = RelayParser.ArgWithAttrContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 360
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==RelayParser.T__2:
                    self.state = 355
                    self.var()
                    self.state = 356
                    self.match(RelayParser.T__4)
                    self.state = 362
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 363
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
            self.state = 374
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__2:
                self.state = 366
                self.var()
                self.state = 371
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==RelayParser.T__4:
                    self.state = 367
                    self.match(RelayParser.T__4)
                    self.state = 368
                    self.var()
                    self.state = 373
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
            self.state = 376
            self.localVar()
            self.state = 379
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RelayParser.T__24:
                self.state = 377
                self.match(RelayParser.T__24)
                self.state = 378
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
            self.state = 381
            self.attr()
            self.state = 386
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==RelayParser.T__4:
                self.state = 382
                self.match(RelayParser.T__4)
                self.state = 383
                self.attr()
                self.state = 388
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
            self.state = 389
            self.match(RelayParser.CNAME)
            self.state = 390
            self.match(RelayParser.T__14)
            self.state = 391
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
            self.state = 440
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,46,self._ctx)
            if la_ == 1:
                localctx = RelayParser.TupleTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 393
                self.match(RelayParser.T__5)
                self.state = 394
                self.match(RelayParser.T__6)
                pass

            elif la_ == 2:
                localctx = RelayParser.TupleTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 395
                self.match(RelayParser.T__5)
                self.state = 396
                self.typeExpr()
                self.state = 397
                self.match(RelayParser.T__4)
                self.state = 398
                self.match(RelayParser.T__6)
                pass

            elif la_ == 3:
                localctx = RelayParser.TupleTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 400
                self.match(RelayParser.T__5)
                self.state = 401
                self.typeExpr()
                self.state = 404 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 402
                    self.match(RelayParser.T__4)
                    self.state = 403
                    self.typeExpr()
                    self.state = 406 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (_la==RelayParser.T__4):
                        break

                self.state = 408
                self.match(RelayParser.T__6)
                pass

            elif la_ == 4:
                localctx = RelayParser.TypeCallTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 410
                self.generalIdent()
                self.state = 411
                self.typeParamList()
                pass

            elif la_ == 5:
                localctx = RelayParser.TypeIdentTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 413
                self.generalIdent()
                pass

            elif la_ == 6:
                localctx = RelayParser.TensorTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 414
                self.match(RelayParser.T__25)
                self.state = 415
                self.match(RelayParser.T__7)
                self.state = 416
                self.shapeList()
                self.state = 417
                self.match(RelayParser.T__4)
                self.state = 418
                self.typeExpr()
                self.state = 419
                self.match(RelayParser.T__8)
                pass

            elif la_ == 7:
                localctx = RelayParser.FuncTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 421
                self.match(RelayParser.T__17)
                self.state = 423
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==RelayParser.T__7:
                    self.state = 422
                    self.typeParamList()


                self.state = 425
                self.match(RelayParser.T__5)
                self.state = 434
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RelayParser.T__3) | (1 << RelayParser.T__5) | (1 << RelayParser.T__17) | (1 << RelayParser.T__25) | (1 << RelayParser.CNAME))) != 0):
                    self.state = 426
                    self.typeExpr()
                    self.state = 431
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la==RelayParser.T__4:
                        self.state = 427
                        self.match(RelayParser.T__4)
                        self.state = 428
                        self.typeExpr()
                        self.state = 433
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)



                self.state = 436
                self.match(RelayParser.T__6)
                self.state = 437
                self.match(RelayParser.T__18)
                self.state = 438
                self.typeExpr()
                pass

            elif la_ == 8:
                localctx = RelayParser.IncompleteTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 8)
                self.state = 439
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

        def generalIdent(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RelayParser.GeneralIdentContext)
            else:
                return self.getTypedRuleContext(RelayParser.GeneralIdentContext,i)


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
            self.state = 442
            self.match(RelayParser.T__7)
            self.state = 443
            self.generalIdent()
            self.state = 448
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==RelayParser.T__4:
                self.state = 444
                self.match(RelayParser.T__4)
                self.state = 445
                self.generalIdent()
                self.state = 450
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 451
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
            self.state = 466
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,49,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 453
                self.match(RelayParser.T__5)
                self.state = 454
                self.match(RelayParser.T__6)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 455
                self.match(RelayParser.T__5)
                self.state = 456
                self.shape()
                self.state = 459 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 457
                    self.match(RelayParser.T__4)
                    self.state = 458
                    self.shape()
                    self.state = 461 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (_la==RelayParser.T__4):
                        break

                self.state = 463
                self.match(RelayParser.T__6)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 465
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
            self.state = 468
            self.match(RelayParser.T__26)
            self.state = 469
            self.match(RelayParser.T__7)
            self.state = 470
            self.match(RelayParser.CNAME)
            self.state = 471
            self.match(RelayParser.T__8)
            self.state = 472
            self.match(RelayParser.T__7)
            self.state = 473
            self.match(RelayParser.NAT)
            self.state = 474
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
            self.state = 482
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.T__26]:
                localctx = RelayParser.MetaShapeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 476
                self.meta()
                pass
            elif token in [RelayParser.T__5]:
                localctx = RelayParser.ParensShapeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 477
                self.match(RelayParser.T__5)
                self.state = 478
                self.shape()
                self.state = 479
                self.match(RelayParser.T__6)
                pass
            elif token in [RelayParser.NAT]:
                localctx = RelayParser.IntShapeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 481
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
            self.state = 484
            self.match(RelayParser.T__11)
            self.state = 485
            self.expr(0)
            self.state = 486
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
            self.state = 491
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [RelayParser.FLOAT]:
                localctx = RelayParser.ScalarFloatContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 488
                self.match(RelayParser.FLOAT)
                pass
            elif token in [RelayParser.NAT]:
                localctx = RelayParser.ScalarIntContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 489
                self.match(RelayParser.NAT)
                pass
            elif token in [RelayParser.BOOL_LIT]:
                localctx = RelayParser.ScalarBoolContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 490
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
            self.state = 497
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,52,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 493
                self.generalIdent()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 494
                self.globalVar()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 495
                self.localVar()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 496
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
         




