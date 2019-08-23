// Generated from Relay.g4 by ANTLR 4.7.2
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class RelayLexer extends Lexer {
	static { RuntimeMetaData.checkVersion("4.7.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, T__6=7, T__7=8, T__8=9, 
		T__9=10, T__10=11, T__11=12, T__12=13, T__13=14, T__14=15, T__15=16, T__16=17, 
		T__17=18, T__18=19, T__19=20, T__20=21, T__21=22, T__22=23, T__23=24, 
		SEMVER=25, COMMENT=26, WS=27, LINE_COMMENT=28, QUOTED_STRING=29, MUL=30, 
		DIV=31, ADD=32, SUB=33, LT=34, GT=35, LE=36, GE=37, EQ=38, NE=39, BOOL_LIT=40, 
		CNAME=41, GLOBAL_VAR=42, LOCAL_VAR=43, GRAPH_VAR=44, DATATYPE=45, FLOAT=46, 
		NAT=47, METADATA=48;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6", "T__7", "T__8", 
			"T__9", "T__10", "T__11", "T__12", "T__13", "T__14", "T__15", "T__16", 
			"T__17", "T__18", "T__19", "T__20", "T__21", "T__22", "T__23", "SEMVER", 
			"COMMENT", "WS", "LINE_COMMENT", "ESCAPED_QUOTE", "QUOTED_STRING", "MUL", 
			"DIV", "ADD", "SUB", "LT", "GT", "LE", "GE", "EQ", "NE", "BOOL_LIT", 
			"CNAME", "GLOBAL_VAR", "LOCAL_VAR", "GRAPH_VAR", "DATATYPE", "PREFLOAT", 
			"FLOAT", "NAT", "EXP", "LETTER", "DIGIT", "METADATA"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "','", "'('", "')'", "'{'", "'}'", "'.'", "'['", "']'", "'if'", 
			"'else'", "'let'", "'='", "';'", "';;'", "'fn'", "'->'", "'def'", "'type'", 
			"'|'", "', '", "':'", "'Tensor'", "'_'", "'meta'", "'v0.0.3'", null, 
			null, null, null, "'*'", "'/'", "'+'", "'-'", "'<'", "'>'", "'<='", "'>='", 
			"'=='", "'!='", null, null, null, null, null, "'int64'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, "SEMVER", "COMMENT", "WS", "LINE_COMMENT", "QUOTED_STRING", "MUL", 
			"DIV", "ADD", "SUB", "LT", "GT", "LE", "GE", "EQ", "NE", "BOOL_LIT", 
			"CNAME", "GLOBAL_VAR", "LOCAL_VAR", "GRAPH_VAR", "DATATYPE", "FLOAT", 
			"NAT", "METADATA"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}


	public RelayLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "Relay.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\62\u015a\b\1\4\2"+
		"\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4"+
		"\13\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22"+
		"\t\22\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31"+
		"\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t"+
		" \4!\t!\4\"\t\"\4#\t#\4$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t"+
		"+\4,\t,\4-\t-\4.\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63\t\63\4\64"+
		"\t\64\4\65\t\65\4\66\t\66\3\2\3\2\3\3\3\3\3\4\3\4\3\5\3\5\3\6\3\6\3\7"+
		"\3\7\3\b\3\b\3\t\3\t\3\n\3\n\3\n\3\13\3\13\3\13\3\13\3\13\3\f\3\f\3\f"+
		"\3\f\3\r\3\r\3\16\3\16\3\17\3\17\3\17\3\20\3\20\3\20\3\21\3\21\3\21\3"+
		"\22\3\22\3\22\3\22\3\23\3\23\3\23\3\23\3\23\3\24\3\24\3\25\3\25\3\25\3"+
		"\26\3\26\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\30\3\30\3\31\3\31\3\31\3"+
		"\31\3\31\3\32\3\32\3\32\3\32\3\32\3\32\3\32\3\33\3\33\3\33\3\33\3\33\7"+
		"\33\u00c1\n\33\f\33\16\33\u00c4\13\33\3\33\3\33\3\33\3\33\3\33\3\34\6"+
		"\34\u00cc\n\34\r\34\16\34\u00cd\3\34\3\34\3\35\3\35\3\35\3\35\7\35\u00d6"+
		"\n\35\f\35\16\35\u00d9\13\35\3\35\3\35\3\35\3\35\3\36\3\36\3\36\3\37\3"+
		"\37\3\37\7\37\u00e5\n\37\f\37\16\37\u00e8\13\37\3\37\3\37\3 \3 \3!\3!"+
		"\3\"\3\"\3#\3#\3$\3$\3%\3%\3&\3&\3&\3\'\3\'\3\'\3(\3(\3(\3)\3)\3)\3*\3"+
		"*\3*\3*\3*\3*\3*\3*\3*\5*\u010d\n*\3+\3+\5+\u0111\n+\3+\3+\3+\7+\u0116"+
		"\n+\f+\16+\u0119\13+\3+\3+\7+\u011d\n+\f+\16+\u0120\13+\3,\3,\3,\3-\3"+
		"-\3-\3.\3.\3.\3/\3/\3/\3/\3/\3/\3\60\3\60\3\60\5\60\u0134\n\60\3\60\5"+
		"\60\u0137\n\60\3\61\3\61\3\61\3\62\6\62\u013d\n\62\r\62\16\62\u013e\3"+
		"\63\3\63\5\63\u0143\n\63\3\63\3\63\3\64\3\64\3\65\3\65\3\66\3\66\3\66"+
		"\3\66\3\66\3\66\3\66\3\66\3\66\3\66\3\66\7\66\u0156\n\66\f\66\16\66\u0159"+
		"\13\66\5\u00c2\u00d7\u00e6\2\67\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23"+
		"\13\25\f\27\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26+\27-\30/\31"+
		"\61\32\63\33\65\34\67\359\36;\2=\37? A!C\"E#G$I%K&M\'O(Q)S*U+W,Y-[.]/"+
		"_\2a\60c\61e\2g\2i\2k\62\3\2\b\5\2\13\f\17\17\"\"\4\2\f\f\17\17\4\2GG"+
		"gg\4\2--//\4\2C\\c|\3\2\62;\2\u0165\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2"+
		"\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3\2\2\2\2\23"+
		"\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2\2\2\2\35\3\2"+
		"\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3\2\2\2\2)\3\2"+
		"\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3\2\2\2\2\65\3"+
		"\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2=\3\2\2\2\2?\3\2\2\2\2A\3\2\2\2\2C\3\2"+
		"\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3\2\2\2\2K\3\2\2\2\2M\3\2\2\2\2O\3\2\2\2"+
		"\2Q\3\2\2\2\2S\3\2\2\2\2U\3\2\2\2\2W\3\2\2\2\2Y\3\2\2\2\2[\3\2\2\2\2]"+
		"\3\2\2\2\2a\3\2\2\2\2c\3\2\2\2\2k\3\2\2\2\3m\3\2\2\2\5o\3\2\2\2\7q\3\2"+
		"\2\2\ts\3\2\2\2\13u\3\2\2\2\rw\3\2\2\2\17y\3\2\2\2\21{\3\2\2\2\23}\3\2"+
		"\2\2\25\u0080\3\2\2\2\27\u0085\3\2\2\2\31\u0089\3\2\2\2\33\u008b\3\2\2"+
		"\2\35\u008d\3\2\2\2\37\u0090\3\2\2\2!\u0093\3\2\2\2#\u0096\3\2\2\2%\u009a"+
		"\3\2\2\2\'\u009f\3\2\2\2)\u00a1\3\2\2\2+\u00a4\3\2\2\2-\u00a6\3\2\2\2"+
		"/\u00ad\3\2\2\2\61\u00af\3\2\2\2\63\u00b4\3\2\2\2\65\u00bb\3\2\2\2\67"+
		"\u00cb\3\2\2\29\u00d1\3\2\2\2;\u00de\3\2\2\2=\u00e1\3\2\2\2?\u00eb\3\2"+
		"\2\2A\u00ed\3\2\2\2C\u00ef\3\2\2\2E\u00f1\3\2\2\2G\u00f3\3\2\2\2I\u00f5"+
		"\3\2\2\2K\u00f7\3\2\2\2M\u00fa\3\2\2\2O\u00fd\3\2\2\2Q\u0100\3\2\2\2S"+
		"\u010c\3\2\2\2U\u0110\3\2\2\2W\u0121\3\2\2\2Y\u0124\3\2\2\2[\u0127\3\2"+
		"\2\2]\u012a\3\2\2\2_\u0130\3\2\2\2a\u0138\3\2\2\2c\u013c\3\2\2\2e\u0140"+
		"\3\2\2\2g\u0146\3\2\2\2i\u0148\3\2\2\2k\u014a\3\2\2\2mn\7.\2\2n\4\3\2"+
		"\2\2op\7*\2\2p\6\3\2\2\2qr\7+\2\2r\b\3\2\2\2st\7}\2\2t\n\3\2\2\2uv\7\177"+
		"\2\2v\f\3\2\2\2wx\7\60\2\2x\16\3\2\2\2yz\7]\2\2z\20\3\2\2\2{|\7_\2\2|"+
		"\22\3\2\2\2}~\7k\2\2~\177\7h\2\2\177\24\3\2\2\2\u0080\u0081\7g\2\2\u0081"+
		"\u0082\7n\2\2\u0082\u0083\7u\2\2\u0083\u0084\7g\2\2\u0084\26\3\2\2\2\u0085"+
		"\u0086\7n\2\2\u0086\u0087\7g\2\2\u0087\u0088\7v\2\2\u0088\30\3\2\2\2\u0089"+
		"\u008a\7?\2\2\u008a\32\3\2\2\2\u008b\u008c\7=\2\2\u008c\34\3\2\2\2\u008d"+
		"\u008e\7=\2\2\u008e\u008f\7=\2\2\u008f\36\3\2\2\2\u0090\u0091\7h\2\2\u0091"+
		"\u0092\7p\2\2\u0092 \3\2\2\2\u0093\u0094\7/\2\2\u0094\u0095\7@\2\2\u0095"+
		"\"\3\2\2\2\u0096\u0097\7f\2\2\u0097\u0098\7g\2\2\u0098\u0099\7h\2\2\u0099"+
		"$\3\2\2\2\u009a\u009b\7v\2\2\u009b\u009c\7{\2\2\u009c\u009d\7r\2\2\u009d"+
		"\u009e\7g\2\2\u009e&\3\2\2\2\u009f\u00a0\7~\2\2\u00a0(\3\2\2\2\u00a1\u00a2"+
		"\7.\2\2\u00a2\u00a3\7\"\2\2\u00a3*\3\2\2\2\u00a4\u00a5\7<\2\2\u00a5,\3"+
		"\2\2\2\u00a6\u00a7\7V\2\2\u00a7\u00a8\7g\2\2\u00a8\u00a9\7p\2\2\u00a9"+
		"\u00aa\7u\2\2\u00aa\u00ab\7q\2\2\u00ab\u00ac\7t\2\2\u00ac.\3\2\2\2\u00ad"+
		"\u00ae\7a\2\2\u00ae\60\3\2\2\2\u00af\u00b0\7o\2\2\u00b0\u00b1\7g\2\2\u00b1"+
		"\u00b2\7v\2\2\u00b2\u00b3\7c\2\2\u00b3\62\3\2\2\2\u00b4\u00b5\7x\2\2\u00b5"+
		"\u00b6\7\62\2\2\u00b6\u00b7\7\60\2\2\u00b7\u00b8\7\62\2\2\u00b8\u00b9"+
		"\7\60\2\2\u00b9\u00ba\7\65\2\2\u00ba\64\3\2\2\2\u00bb\u00bc\7\61\2\2\u00bc"+
		"\u00bd\7,\2\2\u00bd\u00c2\3\2\2\2\u00be\u00c1\5\65\33\2\u00bf\u00c1\13"+
		"\2\2\2\u00c0\u00be\3\2\2\2\u00c0\u00bf\3\2\2\2\u00c1\u00c4\3\2\2\2\u00c2"+
		"\u00c3\3\2\2\2\u00c2\u00c0\3\2\2\2\u00c3\u00c5\3\2\2\2\u00c4\u00c2\3\2"+
		"\2\2\u00c5\u00c6\7,\2\2\u00c6\u00c7\7\61\2\2\u00c7\u00c8\3\2\2\2\u00c8"+
		"\u00c9\b\33\2\2\u00c9\66\3\2\2\2\u00ca\u00cc\t\2\2\2\u00cb\u00ca\3\2\2"+
		"\2\u00cc\u00cd\3\2\2\2\u00cd\u00cb\3\2\2\2\u00cd\u00ce\3\2\2\2\u00ce\u00cf"+
		"\3\2\2\2\u00cf\u00d0\b\34\2\2\u00d08\3\2\2\2\u00d1\u00d2\7\61\2\2\u00d2"+
		"\u00d3\7\61\2\2\u00d3\u00d7\3\2\2\2\u00d4\u00d6\13\2\2\2\u00d5\u00d4\3"+
		"\2\2\2\u00d6\u00d9\3\2\2\2\u00d7\u00d8\3\2\2\2\u00d7\u00d5\3\2\2\2\u00d8"+
		"\u00da\3\2\2\2\u00d9\u00d7\3\2\2\2\u00da\u00db\7\f\2\2\u00db\u00dc\3\2"+
		"\2\2\u00dc\u00dd\b\35\2\2\u00dd:\3\2\2\2\u00de\u00df\7^\2\2\u00df\u00e0"+
		"\7$\2\2\u00e0<\3\2\2\2\u00e1\u00e6\7$\2\2\u00e2\u00e5\5;\36\2\u00e3\u00e5"+
		"\n\3\2\2\u00e4\u00e2\3\2\2\2\u00e4\u00e3\3\2\2\2\u00e5\u00e8\3\2\2\2\u00e6"+
		"\u00e7\3\2\2\2\u00e6\u00e4\3\2\2\2\u00e7\u00e9\3\2\2\2\u00e8\u00e6\3\2"+
		"\2\2\u00e9\u00ea\7$\2\2\u00ea>\3\2\2\2\u00eb\u00ec\7,\2\2\u00ec@\3\2\2"+
		"\2\u00ed\u00ee\7\61\2\2\u00eeB\3\2\2\2\u00ef\u00f0\7-\2\2\u00f0D\3\2\2"+
		"\2\u00f1\u00f2\7/\2\2\u00f2F\3\2\2\2\u00f3\u00f4\7>\2\2\u00f4H\3\2\2\2"+
		"\u00f5\u00f6\7@\2\2\u00f6J\3\2\2\2\u00f7\u00f8\7>\2\2\u00f8\u00f9\7?\2"+
		"\2\u00f9L\3\2\2\2\u00fa\u00fb\7@\2\2\u00fb\u00fc\7?\2\2\u00fcN\3\2\2\2"+
		"\u00fd\u00fe\7?\2\2\u00fe\u00ff\7?\2\2\u00ffP\3\2\2\2\u0100\u0101\7#\2"+
		"\2\u0101\u0102\7?\2\2\u0102R\3\2\2\2\u0103\u0104\7V\2\2\u0104\u0105\7"+
		"t\2\2\u0105\u0106\7w\2\2\u0106\u010d\7g\2\2\u0107\u0108\7H\2\2\u0108\u0109"+
		"\7c\2\2\u0109\u010a\7n\2\2\u010a\u010b\7u\2\2\u010b\u010d\7g\2\2\u010c"+
		"\u0103\3\2\2\2\u010c\u0107\3\2\2\2\u010dT\3\2\2\2\u010e\u0111\7a\2\2\u010f"+
		"\u0111\5g\64\2\u0110\u010e\3\2\2\2\u0110\u010f\3\2\2\2\u0111\u0117\3\2"+
		"\2\2\u0112\u0116\7a\2\2\u0113\u0116\5g\64\2\u0114\u0116\5i\65\2\u0115"+
		"\u0112\3\2\2\2\u0115\u0113\3\2\2\2\u0115\u0114\3\2\2\2\u0116\u0119\3\2"+
		"\2\2\u0117\u0115\3\2\2\2\u0117\u0118\3\2\2\2\u0118\u011e\3\2\2\2\u0119"+
		"\u0117\3\2\2\2\u011a\u011b\7\60\2\2\u011b\u011d\5U+\2\u011c\u011a\3\2"+
		"\2\2\u011d\u0120\3\2\2\2\u011e\u011c\3\2\2\2\u011e\u011f\3\2\2\2\u011f"+
		"V\3\2\2\2\u0120\u011e\3\2\2\2\u0121\u0122\7B\2\2\u0122\u0123\5U+\2\u0123"+
		"X\3\2\2\2\u0124\u0125\7\'\2\2\u0125\u0126\5U+\2\u0126Z\3\2\2\2\u0127\u0128"+
		"\7\'\2\2\u0128\u0129\5c\62\2\u0129\\\3\2\2\2\u012a\u012b\7k\2\2\u012b"+
		"\u012c\7p\2\2\u012c\u012d\7v\2\2\u012d\u012e\78\2\2\u012e\u012f\7\66\2"+
		"\2\u012f^\3\2\2\2\u0130\u0133\5c\62\2\u0131\u0132\7\60\2\2\u0132\u0134"+
		"\5c\62\2\u0133\u0131\3\2\2\2\u0133\u0134\3\2\2\2\u0134\u0136\3\2\2\2\u0135"+
		"\u0137\5e\63\2\u0136\u0135\3\2\2\2\u0136\u0137\3\2\2\2\u0137`\3\2\2\2"+
		"\u0138\u0139\5_\60\2\u0139\u013a\7h\2\2\u013ab\3\2\2\2\u013b\u013d\5i"+
		"\65\2\u013c\u013b\3\2\2\2\u013d\u013e\3\2\2\2\u013e\u013c\3\2\2\2\u013e"+
		"\u013f\3\2\2\2\u013fd\3\2\2\2\u0140\u0142\t\4\2\2\u0141\u0143\t\5\2\2"+
		"\u0142\u0141\3\2\2\2\u0142\u0143\3\2\2\2\u0143\u0144\3\2\2\2\u0144\u0145"+
		"\5c\62\2\u0145f\3\2\2\2\u0146\u0147\t\6\2\2\u0147h\3\2\2\2\u0148\u0149"+
		"\t\7\2\2\u0149j\3\2\2\2\u014a\u014b\7O\2\2\u014b\u014c\7G\2\2\u014c\u014d"+
		"\7V\2\2\u014d\u014e\7C\2\2\u014e\u014f\7F\2\2\u014f\u0150\7C\2\2\u0150"+
		"\u0151\7V\2\2\u0151\u0152\7C\2\2\u0152\u0153\7<\2\2\u0153\u0157\3\2\2"+
		"\2\u0154\u0156\13\2\2\2\u0155\u0154\3\2\2\2\u0156\u0159\3\2\2\2\u0157"+
		"\u0155\3\2\2\2\u0157\u0158\3\2\2\2\u0158l\3\2\2\2\u0159\u0157\3\2\2\2"+
		"\23\2\u00c0\u00c2\u00cd\u00d7\u00e4\u00e6\u010c\u0110\u0115\u0117\u011e"+
		"\u0133\u0136\u013e\u0142\u0157\3\b\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}