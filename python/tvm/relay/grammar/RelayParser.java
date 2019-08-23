// Generated from Relay.g4 by ANTLR 4.7.2
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class RelayParser extends Parser {
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
	public static final int
		RULE_opIdent = 0, RULE_prog = 1, RULE_exprList = 2, RULE_callList = 3, 
		RULE_expr = 4, RULE_func = 5, RULE_defn = 6, RULE_adtDefn = 7, RULE_adtVariant = 8, 
		RULE_argList = 9, RULE_varList = 10, RULE_var = 11, RULE_attrSeq = 12, 
		RULE_attr = 13, RULE_typeParamList = 14, RULE_type_ = 15, RULE_shapeList = 16, 
		RULE_meta = 17, RULE_shape = 18, RULE_typeIdent = 19, RULE_body = 20, 
		RULE_scalar = 21, RULE_ident = 22;
	private static String[] makeRuleNames() {
		return new String[] {
			"opIdent", "prog", "exprList", "callList", "expr", "func", "defn", "adtDefn", 
			"adtVariant", "argList", "varList", "var", "attrSeq", "attr", "typeParamList", 
			"type_", "shapeList", "meta", "shape", "typeIdent", "body", "scalar", 
			"ident"
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

	@Override
	public String getGrammarFileName() { return "Relay.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public RelayParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class OpIdentContext extends ParserRuleContext {
		public TerminalNode CNAME() { return getToken(RelayParser.CNAME, 0); }
		public OpIdentContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_opIdent; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterOpIdent(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitOpIdent(this);
		}
	}

	public final OpIdentContext opIdent() throws RecognitionException {
		OpIdentContext _localctx = new OpIdentContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_opIdent);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(46);
			match(CNAME);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ProgContext extends ParserRuleContext {
		public TerminalNode SEMVER() { return getToken(RelayParser.SEMVER, 0); }
		public TerminalNode EOF() { return getToken(RelayParser.EOF, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode METADATA() { return getToken(RelayParser.METADATA, 0); }
		public List<DefnContext> defn() {
			return getRuleContexts(DefnContext.class);
		}
		public DefnContext defn(int i) {
			return getRuleContext(DefnContext.class,i);
		}
		public ProgContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_prog; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterProg(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitProg(this);
		}
	}

	public final ProgContext prog() throws RecognitionException {
		ProgContext _localctx = new ProgContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_prog);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(48);
			match(SEMVER);
			setState(56);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case EOF:
			case T__16:
			case T__17:
			case METADATA:
				{
				setState(52);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__16 || _la==T__17) {
					{
					{
					setState(49);
					defn();
					}
					}
					setState(54);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
				break;
			case T__1:
			case T__3:
			case T__6:
			case T__8:
			case T__10:
			case T__14:
			case T__23:
			case QUOTED_STRING:
			case SUB:
			case BOOL_LIT:
			case CNAME:
			case GLOBAL_VAR:
			case LOCAL_VAR:
			case GRAPH_VAR:
			case FLOAT:
			case NAT:
				{
				setState(55);
				expr(0);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(59);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==METADATA) {
				{
				setState(58);
				match(METADATA);
				}
			}

			setState(61);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ExprListContext extends ParserRuleContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public ExprListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_exprList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterExprList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitExprList(this);
		}
	}

	public final ExprListContext exprList() throws RecognitionException {
		ExprListContext _localctx = new ExprListContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_exprList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(71);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if ((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << T__1) | (1L << T__3) | (1L << T__6) | (1L << T__8) | (1L << T__10) | (1L << T__14) | (1L << T__23) | (1L << QUOTED_STRING) | (1L << SUB) | (1L << BOOL_LIT) | (1L << CNAME) | (1L << GLOBAL_VAR) | (1L << LOCAL_VAR) | (1L << GRAPH_VAR) | (1L << FLOAT) | (1L << NAT))) != 0)) {
				{
				setState(63);
				expr(0);
				setState(68);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__0) {
					{
					{
					setState(64);
					match(T__0);
					setState(65);
					expr(0);
					}
					}
					setState(70);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class CallListContext extends ParserRuleContext {
		public CallListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_callList; }
	 
		public CallListContext() { }
		public void copyFrom(CallListContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class CallWithAttrContext extends CallListContext {
		public AttrSeqContext attrSeq() {
			return getRuleContext(AttrSeqContext.class,0);
		}
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public CallWithAttrContext(CallListContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterCallWithAttr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitCallWithAttr(this);
		}
	}
	public static class CallNoAttrContext extends CallListContext {
		public ExprListContext exprList() {
			return getRuleContext(ExprListContext.class,0);
		}
		public CallNoAttrContext(CallListContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterCallNoAttr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitCallNoAttr(this);
		}
	}

	public final CallListContext callList() throws RecognitionException {
		CallListContext _localctx = new CallListContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_callList);
		try {
			int _alt;
			setState(83);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,6,_ctx) ) {
			case 1:
				_localctx = new CallNoAttrContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(73);
				exprList();
				}
				break;
			case 2:
				_localctx = new CallWithAttrContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(79);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,5,_ctx);
				while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
					if ( _alt==1 ) {
						{
						{
						setState(74);
						expr(0);
						setState(75);
						match(T__0);
						}
						} 
					}
					setState(81);
					_errHandler.sync(this);
					_alt = getInterpreter().adaptivePredict(_input,5,_ctx);
				}
				setState(82);
				attrSeq();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ExprContext extends ParserRuleContext {
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
	 
		public ExprContext() { }
		public void copyFrom(ExprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class FuncExprContext extends ExprContext {
		public FuncContext func() {
			return getRuleContext(FuncContext.class,0);
		}
		public FuncExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterFuncExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitFuncExpr(this);
		}
	}
	public static class MetaExprContext extends ExprContext {
		public MetaContext meta() {
			return getRuleContext(MetaContext.class,0);
		}
		public MetaExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterMetaExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitMetaExpr(this);
		}
	}
	public static class TensorContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TensorContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterTensor(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitTensor(this);
		}
	}
	public static class GraphContext extends ExprContext {
		public TerminalNode GRAPH_VAR() { return getToken(RelayParser.GRAPH_VAR, 0); }
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public GraphContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterGraph(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitGraph(this);
		}
	}
	public static class IdentExprContext extends ExprContext {
		public IdentContext ident() {
			return getRuleContext(IdentContext.class,0);
		}
		public IdentExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterIdentExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitIdentExpr(this);
		}
	}
	public static class StringExprContext extends ExprContext {
		public TerminalNode QUOTED_STRING() { return getToken(RelayParser.QUOTED_STRING, 0); }
		public StringExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterStringExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitStringExpr(this);
		}
	}
	public static class CallContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public CallListContext callList() {
			return getRuleContext(CallListContext.class,0);
		}
		public CallContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterCall(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitCall(this);
		}
	}
	public static class NegContext extends ExprContext {
		public TerminalNode SUB() { return getToken(RelayParser.SUB, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public NegContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterNeg(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitNeg(this);
		}
	}
	public static class TupleContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TupleContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterTuple(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitTuple(this);
		}
	}
	public static class ParenContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ParenContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterParen(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitParen(this);
		}
	}
	public static class ScalarExprContext extends ExprContext {
		public ScalarContext scalar() {
			return getRuleContext(ScalarContext.class,0);
		}
		public ScalarExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterScalarExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitScalarExpr(this);
		}
	}
	public static class LetContext extends ExprContext {
		public VarContext var() {
			return getRuleContext(VarContext.class,0);
		}
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public LetContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterLet(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitLet(this);
		}
	}
	public static class ProjectionContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode NAT() { return getToken(RelayParser.NAT, 0); }
		public ProjectionContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterProjection(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitProjection(this);
		}
	}
	public static class IfElseContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public List<BodyContext> body() {
			return getRuleContexts(BodyContext.class);
		}
		public BodyContext body(int i) {
			return getRuleContext(BodyContext.class,i);
		}
		public IfElseContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterIfElse(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitIfElse(this);
		}
	}
	public static class BinOpContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode MUL() { return getToken(RelayParser.MUL, 0); }
		public TerminalNode DIV() { return getToken(RelayParser.DIV, 0); }
		public TerminalNode ADD() { return getToken(RelayParser.ADD, 0); }
		public TerminalNode SUB() { return getToken(RelayParser.SUB, 0); }
		public TerminalNode LT() { return getToken(RelayParser.LT, 0); }
		public TerminalNode GT() { return getToken(RelayParser.GT, 0); }
		public TerminalNode LE() { return getToken(RelayParser.LE, 0); }
		public TerminalNode GE() { return getToken(RelayParser.GE, 0); }
		public TerminalNode EQ() { return getToken(RelayParser.EQ, 0); }
		public TerminalNode NE() { return getToken(RelayParser.NE, 0); }
		public BinOpContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterBinOp(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitBinOp(this);
		}
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 8;
		enterRecursionRule(_localctx, 8, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(151);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,10,_ctx) ) {
			case 1:
				{
				_localctx = new ParenContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(86);
				match(T__1);
				setState(87);
				expr(0);
				setState(88);
				match(T__2);
				}
				break;
			case 2:
				{
				_localctx = new ParenContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(90);
				match(T__3);
				setState(91);
				expr(0);
				setState(92);
				match(T__4);
				}
				break;
			case 3:
				{
				_localctx = new NegContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(94);
				match(SUB);
				setState(95);
				expr(19);
				}
				break;
			case 4:
				{
				_localctx = new FuncExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(96);
				func();
				}
				break;
			case 5:
				{
				_localctx = new TupleContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(97);
				match(T__1);
				setState(98);
				match(T__2);
				}
				break;
			case 6:
				{
				_localctx = new TupleContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(99);
				match(T__1);
				setState(100);
				expr(0);
				setState(101);
				match(T__0);
				setState(102);
				match(T__2);
				}
				break;
			case 7:
				{
				_localctx = new TupleContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(104);
				match(T__1);
				setState(105);
				expr(0);
				setState(108); 
				_errHandler.sync(this);
				_la = _input.LA(1);
				do {
					{
					{
					setState(106);
					match(T__0);
					setState(107);
					expr(0);
					}
					}
					setState(110); 
					_errHandler.sync(this);
					_la = _input.LA(1);
				} while ( _la==T__0 );
				setState(112);
				match(T__2);
				}
				break;
			case 8:
				{
				_localctx = new TensorContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(114);
				match(T__6);
				setState(123);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << T__1) | (1L << T__3) | (1L << T__6) | (1L << T__8) | (1L << T__10) | (1L << T__14) | (1L << T__23) | (1L << QUOTED_STRING) | (1L << SUB) | (1L << BOOL_LIT) | (1L << CNAME) | (1L << GLOBAL_VAR) | (1L << LOCAL_VAR) | (1L << GRAPH_VAR) | (1L << FLOAT) | (1L << NAT))) != 0)) {
					{
					setState(115);
					expr(0);
					setState(120);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__0) {
						{
						{
						setState(116);
						match(T__0);
						setState(117);
						expr(0);
						}
						}
						setState(122);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				setState(125);
				match(T__7);
				}
				break;
			case 9:
				{
				_localctx = new IfElseContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(126);
				match(T__8);
				setState(127);
				match(T__1);
				setState(128);
				expr(0);
				setState(129);
				match(T__2);
				setState(130);
				body();
				setState(131);
				match(T__9);
				setState(132);
				body();
				}
				break;
			case 10:
				{
				_localctx = new LetContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(134);
				match(T__10);
				setState(135);
				var();
				setState(136);
				match(T__11);
				setState(137);
				expr(0);
				setState(138);
				match(T__12);
				setState(139);
				expr(7);
				}
				break;
			case 11:
				{
				_localctx = new GraphContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(141);
				match(GRAPH_VAR);
				setState(142);
				match(T__11);
				setState(143);
				expr(0);
				setState(144);
				match(T__12);
				setState(145);
				expr(5);
				}
				break;
			case 12:
				{
				_localctx = new IdentExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(147);
				ident();
				}
				break;
			case 13:
				{
				_localctx = new ScalarExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(148);
				scalar();
				}
				break;
			case 14:
				{
				_localctx = new MetaExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(149);
				meta();
				}
				break;
			case 15:
				{
				_localctx = new StringExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(150);
				match(QUOTED_STRING);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(178);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,12,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(176);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,11,_ctx) ) {
					case 1:
						{
						_localctx = new BinOpContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(153);
						if (!(precpred(_ctx, 18))) throw new FailedPredicateException(this, "precpred(_ctx, 18)");
						setState(154);
						((BinOpContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==MUL || _la==DIV) ) {
							((BinOpContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(155);
						expr(19);
						}
						break;
					case 2:
						{
						_localctx = new BinOpContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(156);
						if (!(precpred(_ctx, 17))) throw new FailedPredicateException(this, "precpred(_ctx, 17)");
						setState(157);
						((BinOpContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==ADD || _la==SUB) ) {
							((BinOpContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(158);
						expr(18);
						}
						break;
					case 3:
						{
						_localctx = new BinOpContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(159);
						if (!(precpred(_ctx, 16))) throw new FailedPredicateException(this, "precpred(_ctx, 16)");
						setState(160);
						((BinOpContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << LT) | (1L << GT) | (1L << LE) | (1L << GE))) != 0)) ) {
							((BinOpContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(161);
						expr(17);
						}
						break;
					case 4:
						{
						_localctx = new BinOpContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(162);
						if (!(precpred(_ctx, 15))) throw new FailedPredicateException(this, "precpred(_ctx, 15)");
						setState(163);
						((BinOpContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==EQ || _la==NE) ) {
							((BinOpContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(164);
						expr(16);
						}
						break;
					case 5:
						{
						_localctx = new LetContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(165);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(166);
						match(T__13);
						setState(167);
						expr(7);
						}
						break;
					case 6:
						{
						_localctx = new CallContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(168);
						if (!(precpred(_ctx, 20))) throw new FailedPredicateException(this, "precpred(_ctx, 20)");
						setState(169);
						match(T__1);
						setState(170);
						callList();
						setState(171);
						match(T__2);
						}
						break;
					case 7:
						{
						_localctx = new ProjectionContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(173);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(174);
						match(T__5);
						setState(175);
						match(NAT);
						}
						break;
					}
					} 
				}
				setState(180);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,12,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class FuncContext extends ParserRuleContext {
		public ArgListContext argList() {
			return getRuleContext(ArgListContext.class,0);
		}
		public BodyContext body() {
			return getRuleContext(BodyContext.class,0);
		}
		public TypeParamListContext typeParamList() {
			return getRuleContext(TypeParamListContext.class,0);
		}
		public Type_Context type_() {
			return getRuleContext(Type_Context.class,0);
		}
		public FuncContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_func; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterFunc(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitFunc(this);
		}
	}

	public final FuncContext func() throws RecognitionException {
		FuncContext _localctx = new FuncContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_func);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(181);
			match(T__14);
			setState(183);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__6) {
				{
				setState(182);
				typeParamList();
				}
			}

			setState(185);
			match(T__1);
			setState(186);
			argList();
			setState(187);
			match(T__2);
			setState(190);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__15) {
				{
				setState(188);
				match(T__15);
				setState(189);
				type_();
				}
			}

			setState(192);
			body();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class DefnContext extends ParserRuleContext {
		public IdentContext ident() {
			return getRuleContext(IdentContext.class,0);
		}
		public ArgListContext argList() {
			return getRuleContext(ArgListContext.class,0);
		}
		public BodyContext body() {
			return getRuleContext(BodyContext.class,0);
		}
		public TypeParamListContext typeParamList() {
			return getRuleContext(TypeParamListContext.class,0);
		}
		public Type_Context type_() {
			return getRuleContext(Type_Context.class,0);
		}
		public AdtDefnContext adtDefn() {
			return getRuleContext(AdtDefnContext.class,0);
		}
		public DefnContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_defn; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterDefn(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitDefn(this);
		}
	}

	public final DefnContext defn() throws RecognitionException {
		DefnContext _localctx = new DefnContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_defn);
		int _la;
		try {
			setState(209);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__16:
				enterOuterAlt(_localctx, 1);
				{
				setState(194);
				match(T__16);
				setState(195);
				ident();
				setState(197);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__6) {
					{
					setState(196);
					typeParamList();
					}
				}

				setState(199);
				match(T__1);
				setState(200);
				argList();
				setState(201);
				match(T__2);
				setState(204);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__15) {
					{
					setState(202);
					match(T__15);
					setState(203);
					type_();
					}
				}

				setState(206);
				body();
				}
				break;
			case T__17:
				enterOuterAlt(_localctx, 2);
				{
				setState(208);
				adtDefn();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AdtDefnContext extends ParserRuleContext {
		public TypeIdentContext typeIdent() {
			return getRuleContext(TypeIdentContext.class,0);
		}
		public List<TerminalNode> LOCAL_VAR() { return getTokens(RelayParser.LOCAL_VAR); }
		public TerminalNode LOCAL_VAR(int i) {
			return getToken(RelayParser.LOCAL_VAR, i);
		}
		public List<AdtVariantContext> adtVariant() {
			return getRuleContexts(AdtVariantContext.class);
		}
		public AdtVariantContext adtVariant(int i) {
			return getRuleContext(AdtVariantContext.class,i);
		}
		public AdtDefnContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_adtDefn; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterAdtDefn(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitAdtDefn(this);
		}
	}

	public final AdtDefnContext adtDefn() throws RecognitionException {
		AdtDefnContext _localctx = new AdtDefnContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_adtDefn);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(211);
			match(T__17);
			setState(212);
			typeIdent();
			setState(223);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__6) {
				{
				setState(213);
				match(T__6);
				setState(214);
				match(LOCAL_VAR);
				setState(219);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__0) {
					{
					{
					setState(215);
					match(T__0);
					setState(216);
					match(LOCAL_VAR);
					}
					}
					setState(221);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(222);
				match(T__7);
				}
			}

			setState(225);
			match(T__11);
			setState(227); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(226);
				adtVariant();
				}
				}
				setState(229); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( _la==T__18 );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AdtVariantContext extends ParserRuleContext {
		public TypeIdentContext typeIdent() {
			return getRuleContext(TypeIdentContext.class,0);
		}
		public List<Type_Context> type_() {
			return getRuleContexts(Type_Context.class);
		}
		public Type_Context type_(int i) {
			return getRuleContext(Type_Context.class,i);
		}
		public AdtVariantContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_adtVariant; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterAdtVariant(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitAdtVariant(this);
		}
	}

	public final AdtVariantContext adtVariant() throws RecognitionException {
		AdtVariantContext _localctx = new AdtVariantContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_adtVariant);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(231);
			match(T__18);
			setState(232);
			typeIdent();
			setState(244);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__1) {
				{
				setState(233);
				match(T__1);
				setState(234);
				type_();
				setState(239);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__19) {
					{
					{
					setState(235);
					match(T__19);
					setState(236);
					type_();
					}
					}
					setState(241);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(242);
				match(T__2);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ArgListContext extends ParserRuleContext {
		public ArgListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_argList; }
	 
		public ArgListContext() { }
		public void copyFrom(ArgListContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class ArgNoAttrContext extends ArgListContext {
		public VarListContext varList() {
			return getRuleContext(VarListContext.class,0);
		}
		public ArgNoAttrContext(ArgListContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterArgNoAttr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitArgNoAttr(this);
		}
	}
	public static class ArgWithAttrContext extends ArgListContext {
		public AttrSeqContext attrSeq() {
			return getRuleContext(AttrSeqContext.class,0);
		}
		public List<VarContext> var() {
			return getRuleContexts(VarContext.class);
		}
		public VarContext var(int i) {
			return getRuleContext(VarContext.class,i);
		}
		public ArgWithAttrContext(ArgListContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterArgWithAttr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitArgWithAttr(this);
		}
	}

	public final ArgListContext argList() throws RecognitionException {
		ArgListContext _localctx = new ArgListContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_argList);
		int _la;
		try {
			setState(256);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,24,_ctx) ) {
			case 1:
				_localctx = new ArgNoAttrContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(246);
				varList();
				}
				break;
			case 2:
				_localctx = new ArgWithAttrContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(252);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==LOCAL_VAR) {
					{
					{
					setState(247);
					var();
					setState(248);
					match(T__0);
					}
					}
					setState(254);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(255);
				attrSeq();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VarListContext extends ParserRuleContext {
		public List<VarContext> var() {
			return getRuleContexts(VarContext.class);
		}
		public VarContext var(int i) {
			return getRuleContext(VarContext.class,i);
		}
		public VarListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_varList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterVarList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitVarList(this);
		}
	}

	public final VarListContext varList() throws RecognitionException {
		VarListContext _localctx = new VarListContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_varList);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(266);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==LOCAL_VAR) {
				{
				setState(258);
				var();
				setState(263);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__0) {
					{
					{
					setState(259);
					match(T__0);
					setState(260);
					var();
					}
					}
					setState(265);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VarContext extends ParserRuleContext {
		public TerminalNode LOCAL_VAR() { return getToken(RelayParser.LOCAL_VAR, 0); }
		public Type_Context type_() {
			return getRuleContext(Type_Context.class,0);
		}
		public VarContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_var; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterVar(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitVar(this);
		}
	}

	public final VarContext var() throws RecognitionException {
		VarContext _localctx = new VarContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_var);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(268);
			match(LOCAL_VAR);
			setState(271);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__20) {
				{
				setState(269);
				match(T__20);
				setState(270);
				type_();
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AttrSeqContext extends ParserRuleContext {
		public List<AttrContext> attr() {
			return getRuleContexts(AttrContext.class);
		}
		public AttrContext attr(int i) {
			return getRuleContext(AttrContext.class,i);
		}
		public AttrSeqContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attrSeq; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterAttrSeq(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitAttrSeq(this);
		}
	}

	public final AttrSeqContext attrSeq() throws RecognitionException {
		AttrSeqContext _localctx = new AttrSeqContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_attrSeq);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(273);
			attr();
			setState(278);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__0) {
				{
				{
				setState(274);
				match(T__0);
				setState(275);
				attr();
				}
				}
				setState(280);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AttrContext extends ParserRuleContext {
		public TerminalNode CNAME() { return getToken(RelayParser.CNAME, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public AttrContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attr; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterAttr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitAttr(this);
		}
	}

	public final AttrContext attr() throws RecognitionException {
		AttrContext _localctx = new AttrContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_attr);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(281);
			match(CNAME);
			setState(282);
			match(T__11);
			setState(283);
			expr(0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class TypeParamListContext extends ParserRuleContext {
		public List<IdentContext> ident() {
			return getRuleContexts(IdentContext.class);
		}
		public IdentContext ident(int i) {
			return getRuleContext(IdentContext.class,i);
		}
		public TypeParamListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_typeParamList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterTypeParamList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitTypeParamList(this);
		}
	}

	public final TypeParamListContext typeParamList() throws RecognitionException {
		TypeParamListContext _localctx = new TypeParamListContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_typeParamList);
		int _la;
		try {
			setState(298);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,30,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(285);
				match(T__6);
				setState(286);
				match(T__7);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(287);
				match(T__6);
				setState(288);
				ident();
				setState(293);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__0) {
					{
					{
					setState(289);
					match(T__0);
					setState(290);
					ident();
					}
					}
					setState(295);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(296);
				match(T__7);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Type_Context extends ParserRuleContext {
		public Type_Context(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_type_; }
	 
		public Type_Context() { }
		public void copyFrom(Type_Context ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class IntTypeContext extends Type_Context {
		public TerminalNode NAT() { return getToken(RelayParser.NAT, 0); }
		public IntTypeContext(Type_Context ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterIntType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitIntType(this);
		}
	}
	public static class TupleTypeContext extends Type_Context {
		public List<Type_Context> type_() {
			return getRuleContexts(Type_Context.class);
		}
		public Type_Context type_(int i) {
			return getRuleContext(Type_Context.class,i);
		}
		public TupleTypeContext(Type_Context ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterTupleType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitTupleType(this);
		}
	}
	public static class TypeIdentTypeContext extends Type_Context {
		public TypeIdentContext typeIdent() {
			return getRuleContext(TypeIdentContext.class,0);
		}
		public TypeIdentTypeContext(Type_Context ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterTypeIdentType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitTypeIdentType(this);
		}
	}
	public static class IncompleteTypeContext extends Type_Context {
		public IncompleteTypeContext(Type_Context ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterIncompleteType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitIncompleteType(this);
		}
	}
	public static class TensorTypeContext extends Type_Context {
		public ShapeListContext shapeList() {
			return getRuleContext(ShapeListContext.class,0);
		}
		public Type_Context type_() {
			return getRuleContext(Type_Context.class,0);
		}
		public TensorTypeContext(Type_Context ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterTensorType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitTensorType(this);
		}
	}
	public static class FuncTypeContext extends Type_Context {
		public List<Type_Context> type_() {
			return getRuleContexts(Type_Context.class);
		}
		public Type_Context type_(int i) {
			return getRuleContext(Type_Context.class,i);
		}
		public TypeParamListContext typeParamList() {
			return getRuleContext(TypeParamListContext.class,0);
		}
		public FuncTypeContext(Type_Context ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterFuncType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitFuncType(this);
		}
	}

	public final Type_Context type_() throws RecognitionException {
		Type_Context _localctx = new Type_Context(_ctx, getState());
		enterRule(_localctx, 30, RULE_type_);
		int _la;
		try {
			setState(345);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,35,_ctx) ) {
			case 1:
				_localctx = new TupleTypeContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(300);
				match(T__1);
				setState(301);
				match(T__2);
				}
				break;
			case 2:
				_localctx = new TupleTypeContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(302);
				match(T__1);
				setState(303);
				type_();
				setState(304);
				match(T__0);
				setState(305);
				match(T__2);
				}
				break;
			case 3:
				_localctx = new TupleTypeContext(_localctx);
				enterOuterAlt(_localctx, 3);
				{
				setState(307);
				match(T__1);
				setState(308);
				type_();
				setState(311); 
				_errHandler.sync(this);
				_la = _input.LA(1);
				do {
					{
					{
					setState(309);
					match(T__0);
					setState(310);
					type_();
					}
					}
					setState(313); 
					_errHandler.sync(this);
					_la = _input.LA(1);
				} while ( _la==T__0 );
				setState(315);
				match(T__2);
				}
				break;
			case 4:
				_localctx = new TypeIdentTypeContext(_localctx);
				enterOuterAlt(_localctx, 4);
				{
				setState(317);
				typeIdent();
				}
				break;
			case 5:
				_localctx = new TensorTypeContext(_localctx);
				enterOuterAlt(_localctx, 5);
				{
				setState(318);
				match(T__21);
				setState(319);
				match(T__6);
				setState(320);
				shapeList();
				setState(321);
				match(T__0);
				setState(322);
				type_();
				setState(323);
				match(T__7);
				}
				break;
			case 6:
				_localctx = new FuncTypeContext(_localctx);
				enterOuterAlt(_localctx, 6);
				{
				setState(325);
				match(T__14);
				setState(327);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==T__6) {
					{
					setState(326);
					typeParamList();
					}
				}

				setState(329);
				match(T__1);
				setState(338);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if ((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << T__1) | (1L << T__14) | (1L << T__21) | (1L << T__22) | (1L << CNAME) | (1L << NAT))) != 0)) {
					{
					setState(330);
					type_();
					setState(335);
					_errHandler.sync(this);
					_la = _input.LA(1);
					while (_la==T__0) {
						{
						{
						setState(331);
						match(T__0);
						setState(332);
						type_();
						}
						}
						setState(337);
						_errHandler.sync(this);
						_la = _input.LA(1);
					}
					}
				}

				setState(340);
				match(T__2);
				setState(341);
				match(T__15);
				setState(342);
				type_();
				}
				break;
			case 7:
				_localctx = new IncompleteTypeContext(_localctx);
				enterOuterAlt(_localctx, 7);
				{
				setState(343);
				match(T__22);
				}
				break;
			case 8:
				_localctx = new IntTypeContext(_localctx);
				enterOuterAlt(_localctx, 8);
				{
				setState(344);
				match(NAT);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ShapeListContext extends ParserRuleContext {
		public List<ShapeContext> shape() {
			return getRuleContexts(ShapeContext.class);
		}
		public ShapeContext shape(int i) {
			return getRuleContext(ShapeContext.class,i);
		}
		public ShapeListContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_shapeList; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterShapeList(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitShapeList(this);
		}
	}

	public final ShapeListContext shapeList() throws RecognitionException {
		ShapeListContext _localctx = new ShapeListContext(_ctx, getState());
		enterRule(_localctx, 32, RULE_shapeList);
		int _la;
		try {
			setState(360);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,37,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(347);
				match(T__1);
				setState(348);
				shape();
				setState(351); 
				_errHandler.sync(this);
				_la = _input.LA(1);
				do {
					{
					{
					setState(349);
					match(T__0);
					setState(350);
					shape();
					}
					}
					setState(353); 
					_errHandler.sync(this);
					_la = _input.LA(1);
				} while ( _la==T__0 );
				setState(355);
				match(T__2);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(357);
				match(T__1);
				setState(358);
				match(T__2);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(359);
				shape();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class MetaContext extends ParserRuleContext {
		public TerminalNode CNAME() { return getToken(RelayParser.CNAME, 0); }
		public TerminalNode NAT() { return getToken(RelayParser.NAT, 0); }
		public MetaContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_meta; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterMeta(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitMeta(this);
		}
	}

	public final MetaContext meta() throws RecognitionException {
		MetaContext _localctx = new MetaContext(_ctx, getState());
		enterRule(_localctx, 34, RULE_meta);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(362);
			match(T__23);
			setState(363);
			match(T__6);
			setState(364);
			match(CNAME);
			setState(365);
			match(T__7);
			setState(366);
			match(T__6);
			setState(367);
			match(NAT);
			setState(368);
			match(T__7);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ShapeContext extends ParserRuleContext {
		public ShapeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_shape; }
	 
		public ShapeContext() { }
		public void copyFrom(ShapeContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class ParensShapeContext extends ShapeContext {
		public ShapeContext shape() {
			return getRuleContext(ShapeContext.class,0);
		}
		public ParensShapeContext(ShapeContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterParensShape(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitParensShape(this);
		}
	}
	public static class MetaShapeContext extends ShapeContext {
		public MetaContext meta() {
			return getRuleContext(MetaContext.class,0);
		}
		public MetaShapeContext(ShapeContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterMetaShape(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitMetaShape(this);
		}
	}
	public static class IntShapeContext extends ShapeContext {
		public TerminalNode NAT() { return getToken(RelayParser.NAT, 0); }
		public IntShapeContext(ShapeContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterIntShape(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitIntShape(this);
		}
	}

	public final ShapeContext shape() throws RecognitionException {
		ShapeContext _localctx = new ShapeContext(_ctx, getState());
		enterRule(_localctx, 36, RULE_shape);
		try {
			setState(376);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__23:
				_localctx = new MetaShapeContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(370);
				meta();
				}
				break;
			case T__1:
				_localctx = new ParensShapeContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(371);
				match(T__1);
				setState(372);
				shape();
				setState(373);
				match(T__2);
				}
				break;
			case NAT:
				_localctx = new IntShapeContext(_localctx);
				enterOuterAlt(_localctx, 3);
				{
				setState(375);
				match(NAT);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class TypeIdentContext extends ParserRuleContext {
		public TerminalNode CNAME() { return getToken(RelayParser.CNAME, 0); }
		public TypeIdentContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_typeIdent; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterTypeIdent(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitTypeIdent(this);
		}
	}

	public final TypeIdentContext typeIdent() throws RecognitionException {
		TypeIdentContext _localctx = new TypeIdentContext(_ctx, getState());
		enterRule(_localctx, 38, RULE_typeIdent);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(378);
			match(CNAME);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class BodyContext extends ParserRuleContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public BodyContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_body; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterBody(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitBody(this);
		}
	}

	public final BodyContext body() throws RecognitionException {
		BodyContext _localctx = new BodyContext(_ctx, getState());
		enterRule(_localctx, 40, RULE_body);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(380);
			match(T__3);
			setState(381);
			expr(0);
			setState(382);
			match(T__4);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ScalarContext extends ParserRuleContext {
		public ScalarContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_scalar; }
	 
		public ScalarContext() { }
		public void copyFrom(ScalarContext ctx) {
			super.copyFrom(ctx);
		}
	}
	public static class ScalarFloatContext extends ScalarContext {
		public TerminalNode FLOAT() { return getToken(RelayParser.FLOAT, 0); }
		public ScalarFloatContext(ScalarContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterScalarFloat(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitScalarFloat(this);
		}
	}
	public static class ScalarBoolContext extends ScalarContext {
		public TerminalNode BOOL_LIT() { return getToken(RelayParser.BOOL_LIT, 0); }
		public ScalarBoolContext(ScalarContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterScalarBool(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitScalarBool(this);
		}
	}
	public static class ScalarIntContext extends ScalarContext {
		public TerminalNode NAT() { return getToken(RelayParser.NAT, 0); }
		public ScalarIntContext(ScalarContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterScalarInt(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitScalarInt(this);
		}
	}

	public final ScalarContext scalar() throws RecognitionException {
		ScalarContext _localctx = new ScalarContext(_ctx, getState());
		enterRule(_localctx, 42, RULE_scalar);
		try {
			setState(387);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case FLOAT:
				_localctx = new ScalarFloatContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(384);
				match(FLOAT);
				}
				break;
			case NAT:
				_localctx = new ScalarIntContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(385);
				match(NAT);
				}
				break;
			case BOOL_LIT:
				_localctx = new ScalarBoolContext(_localctx);
				enterOuterAlt(_localctx, 3);
				{
				setState(386);
				match(BOOL_LIT);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class IdentContext extends ParserRuleContext {
		public OpIdentContext opIdent() {
			return getRuleContext(OpIdentContext.class,0);
		}
		public TerminalNode GLOBAL_VAR() { return getToken(RelayParser.GLOBAL_VAR, 0); }
		public TerminalNode LOCAL_VAR() { return getToken(RelayParser.LOCAL_VAR, 0); }
		public TerminalNode GRAPH_VAR() { return getToken(RelayParser.GRAPH_VAR, 0); }
		public IdentContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_ident; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).enterIdent(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RelayListener ) ((RelayListener)listener).exitIdent(this);
		}
	}

	public final IdentContext ident() throws RecognitionException {
		IdentContext _localctx = new IdentContext(_ctx, getState());
		enterRule(_localctx, 44, RULE_ident);
		try {
			setState(393);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case CNAME:
				enterOuterAlt(_localctx, 1);
				{
				setState(389);
				opIdent();
				}
				break;
			case GLOBAL_VAR:
				enterOuterAlt(_localctx, 2);
				{
				setState(390);
				match(GLOBAL_VAR);
				}
				break;
			case LOCAL_VAR:
				enterOuterAlt(_localctx, 3);
				{
				setState(391);
				match(LOCAL_VAR);
				}
				break;
			case GRAPH_VAR:
				enterOuterAlt(_localctx, 4);
				{
				setState(392);
				match(GRAPH_VAR);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 4:
			return expr_sempred((ExprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 18);
		case 1:
			return precpred(_ctx, 17);
		case 2:
			return precpred(_ctx, 16);
		case 3:
			return precpred(_ctx, 15);
		case 4:
			return precpred(_ctx, 6);
		case 5:
			return precpred(_ctx, 20);
		case 6:
			return precpred(_ctx, 10);
		}
		return true;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\62\u018e\4\2\t\2"+
		"\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\3\2\3\2\3"+
		"\3\3\3\7\3\65\n\3\f\3\16\38\13\3\3\3\5\3;\n\3\3\3\5\3>\n\3\3\3\3\3\3\4"+
		"\3\4\3\4\7\4E\n\4\f\4\16\4H\13\4\5\4J\n\4\3\5\3\5\3\5\3\5\7\5P\n\5\f\5"+
		"\16\5S\13\5\3\5\5\5V\n\5\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3"+
		"\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\6\6o\n\6\r\6\16\6p\3\6"+
		"\3\6\3\6\3\6\3\6\3\6\7\6y\n\6\f\6\16\6|\13\6\5\6~\n\6\3\6\3\6\3\6\3\6"+
		"\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3"+
		"\6\3\6\3\6\3\6\3\6\5\6\u009a\n\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3"+
		"\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\7\6\u00b3\n\6\f"+
		"\6\16\6\u00b6\13\6\3\7\3\7\5\7\u00ba\n\7\3\7\3\7\3\7\3\7\3\7\5\7\u00c1"+
		"\n\7\3\7\3\7\3\b\3\b\3\b\5\b\u00c8\n\b\3\b\3\b\3\b\3\b\3\b\5\b\u00cf\n"+
		"\b\3\b\3\b\3\b\5\b\u00d4\n\b\3\t\3\t\3\t\3\t\3\t\3\t\7\t\u00dc\n\t\f\t"+
		"\16\t\u00df\13\t\3\t\5\t\u00e2\n\t\3\t\3\t\6\t\u00e6\n\t\r\t\16\t\u00e7"+
		"\3\n\3\n\3\n\3\n\3\n\3\n\7\n\u00f0\n\n\f\n\16\n\u00f3\13\n\3\n\3\n\5\n"+
		"\u00f7\n\n\3\13\3\13\3\13\3\13\7\13\u00fd\n\13\f\13\16\13\u0100\13\13"+
		"\3\13\5\13\u0103\n\13\3\f\3\f\3\f\7\f\u0108\n\f\f\f\16\f\u010b\13\f\5"+
		"\f\u010d\n\f\3\r\3\r\3\r\5\r\u0112\n\r\3\16\3\16\3\16\7\16\u0117\n\16"+
		"\f\16\16\16\u011a\13\16\3\17\3\17\3\17\3\17\3\20\3\20\3\20\3\20\3\20\3"+
		"\20\7\20\u0126\n\20\f\20\16\20\u0129\13\20\3\20\3\20\5\20\u012d\n\20\3"+
		"\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\6\21\u013a\n\21"+
		"\r\21\16\21\u013b\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3"+
		"\21\3\21\5\21\u014a\n\21\3\21\3\21\3\21\3\21\7\21\u0150\n\21\f\21\16\21"+
		"\u0153\13\21\5\21\u0155\n\21\3\21\3\21\3\21\3\21\3\21\5\21\u015c\n\21"+
		"\3\22\3\22\3\22\3\22\6\22\u0162\n\22\r\22\16\22\u0163\3\22\3\22\3\22\3"+
		"\22\3\22\5\22\u016b\n\22\3\23\3\23\3\23\3\23\3\23\3\23\3\23\3\23\3\24"+
		"\3\24\3\24\3\24\3\24\3\24\5\24\u017b\n\24\3\25\3\25\3\26\3\26\3\26\3\26"+
		"\3\27\3\27\3\27\5\27\u0186\n\27\3\30\3\30\3\30\3\30\5\30\u018c\n\30\3"+
		"\30\2\3\n\31\2\4\6\b\n\f\16\20\22\24\26\30\32\34\36 \"$&(*,.\2\6\3\2 "+
		"!\3\2\"#\3\2$\'\3\2()\2\u01bc\2\60\3\2\2\2\4\62\3\2\2\2\6I\3\2\2\2\bU"+
		"\3\2\2\2\n\u0099\3\2\2\2\f\u00b7\3\2\2\2\16\u00d3\3\2\2\2\20\u00d5\3\2"+
		"\2\2\22\u00e9\3\2\2\2\24\u0102\3\2\2\2\26\u010c\3\2\2\2\30\u010e\3\2\2"+
		"\2\32\u0113\3\2\2\2\34\u011b\3\2\2\2\36\u012c\3\2\2\2 \u015b\3\2\2\2\""+
		"\u016a\3\2\2\2$\u016c\3\2\2\2&\u017a\3\2\2\2(\u017c\3\2\2\2*\u017e\3\2"+
		"\2\2,\u0185\3\2\2\2.\u018b\3\2\2\2\60\61\7+\2\2\61\3\3\2\2\2\62:\7\33"+
		"\2\2\63\65\5\16\b\2\64\63\3\2\2\2\658\3\2\2\2\66\64\3\2\2\2\66\67\3\2"+
		"\2\2\67;\3\2\2\28\66\3\2\2\29;\5\n\6\2:\66\3\2\2\2:9\3\2\2\2;=\3\2\2\2"+
		"<>\7\62\2\2=<\3\2\2\2=>\3\2\2\2>?\3\2\2\2?@\7\2\2\3@\5\3\2\2\2AF\5\n\6"+
		"\2BC\7\3\2\2CE\5\n\6\2DB\3\2\2\2EH\3\2\2\2FD\3\2\2\2FG\3\2\2\2GJ\3\2\2"+
		"\2HF\3\2\2\2IA\3\2\2\2IJ\3\2\2\2J\7\3\2\2\2KV\5\6\4\2LM\5\n\6\2MN\7\3"+
		"\2\2NP\3\2\2\2OL\3\2\2\2PS\3\2\2\2QO\3\2\2\2QR\3\2\2\2RT\3\2\2\2SQ\3\2"+
		"\2\2TV\5\32\16\2UK\3\2\2\2UQ\3\2\2\2V\t\3\2\2\2WX\b\6\1\2XY\7\4\2\2YZ"+
		"\5\n\6\2Z[\7\5\2\2[\u009a\3\2\2\2\\]\7\6\2\2]^\5\n\6\2^_\7\7\2\2_\u009a"+
		"\3\2\2\2`a\7#\2\2a\u009a\5\n\6\25b\u009a\5\f\7\2cd\7\4\2\2d\u009a\7\5"+
		"\2\2ef\7\4\2\2fg\5\n\6\2gh\7\3\2\2hi\7\5\2\2i\u009a\3\2\2\2jk\7\4\2\2"+
		"kn\5\n\6\2lm\7\3\2\2mo\5\n\6\2nl\3\2\2\2op\3\2\2\2pn\3\2\2\2pq\3\2\2\2"+
		"qr\3\2\2\2rs\7\5\2\2s\u009a\3\2\2\2t}\7\t\2\2uz\5\n\6\2vw\7\3\2\2wy\5"+
		"\n\6\2xv\3\2\2\2y|\3\2\2\2zx\3\2\2\2z{\3\2\2\2{~\3\2\2\2|z\3\2\2\2}u\3"+
		"\2\2\2}~\3\2\2\2~\177\3\2\2\2\177\u009a\7\n\2\2\u0080\u0081\7\13\2\2\u0081"+
		"\u0082\7\4\2\2\u0082\u0083\5\n\6\2\u0083\u0084\7\5\2\2\u0084\u0085\5*"+
		"\26\2\u0085\u0086\7\f\2\2\u0086\u0087\5*\26\2\u0087\u009a\3\2\2\2\u0088"+
		"\u0089\7\r\2\2\u0089\u008a\5\30\r\2\u008a\u008b\7\16\2\2\u008b\u008c\5"+
		"\n\6\2\u008c\u008d\7\17\2\2\u008d\u008e\5\n\6\t\u008e\u009a\3\2\2\2\u008f"+
		"\u0090\7.\2\2\u0090\u0091\7\16\2\2\u0091\u0092\5\n\6\2\u0092\u0093\7\17"+
		"\2\2\u0093\u0094\5\n\6\7\u0094\u009a\3\2\2\2\u0095\u009a\5.\30\2\u0096"+
		"\u009a\5,\27\2\u0097\u009a\5$\23\2\u0098\u009a\7\37\2\2\u0099W\3\2\2\2"+
		"\u0099\\\3\2\2\2\u0099`\3\2\2\2\u0099b\3\2\2\2\u0099c\3\2\2\2\u0099e\3"+
		"\2\2\2\u0099j\3\2\2\2\u0099t\3\2\2\2\u0099\u0080\3\2\2\2\u0099\u0088\3"+
		"\2\2\2\u0099\u008f\3\2\2\2\u0099\u0095\3\2\2\2\u0099\u0096\3\2\2\2\u0099"+
		"\u0097\3\2\2\2\u0099\u0098\3\2\2\2\u009a\u00b4\3\2\2\2\u009b\u009c\f\24"+
		"\2\2\u009c\u009d\t\2\2\2\u009d\u00b3\5\n\6\25\u009e\u009f\f\23\2\2\u009f"+
		"\u00a0\t\3\2\2\u00a0\u00b3\5\n\6\24\u00a1\u00a2\f\22\2\2\u00a2\u00a3\t"+
		"\4\2\2\u00a3\u00b3\5\n\6\23\u00a4\u00a5\f\21\2\2\u00a5\u00a6\t\5\2\2\u00a6"+
		"\u00b3\5\n\6\22\u00a7\u00a8\f\b\2\2\u00a8\u00a9\7\20\2\2\u00a9\u00b3\5"+
		"\n\6\t\u00aa\u00ab\f\26\2\2\u00ab\u00ac\7\4\2\2\u00ac\u00ad\5\b\5\2\u00ad"+
		"\u00ae\7\5\2\2\u00ae\u00b3\3\2\2\2\u00af\u00b0\f\f\2\2\u00b0\u00b1\7\b"+
		"\2\2\u00b1\u00b3\7\61\2\2\u00b2\u009b\3\2\2\2\u00b2\u009e\3\2\2\2\u00b2"+
		"\u00a1\3\2\2\2\u00b2\u00a4\3\2\2\2\u00b2\u00a7\3\2\2\2\u00b2\u00aa\3\2"+
		"\2\2\u00b2\u00af\3\2\2\2\u00b3\u00b6\3\2\2\2\u00b4\u00b2\3\2\2\2\u00b4"+
		"\u00b5\3\2\2\2\u00b5\13\3\2\2\2\u00b6\u00b4\3\2\2\2\u00b7\u00b9\7\21\2"+
		"\2\u00b8\u00ba\5\36\20\2\u00b9\u00b8\3\2\2\2\u00b9\u00ba\3\2\2\2\u00ba"+
		"\u00bb\3\2\2\2\u00bb\u00bc\7\4\2\2\u00bc\u00bd\5\24\13\2\u00bd\u00c0\7"+
		"\5\2\2\u00be\u00bf\7\22\2\2\u00bf\u00c1\5 \21\2\u00c0\u00be\3\2\2\2\u00c0"+
		"\u00c1\3\2\2\2\u00c1\u00c2\3\2\2\2\u00c2\u00c3\5*\26\2\u00c3\r\3\2\2\2"+
		"\u00c4\u00c5\7\23\2\2\u00c5\u00c7\5.\30\2\u00c6\u00c8\5\36\20\2\u00c7"+
		"\u00c6\3\2\2\2\u00c7\u00c8\3\2\2\2\u00c8\u00c9\3\2\2\2\u00c9\u00ca\7\4"+
		"\2\2\u00ca\u00cb\5\24\13\2\u00cb\u00ce\7\5\2\2\u00cc\u00cd\7\22\2\2\u00cd"+
		"\u00cf\5 \21\2\u00ce\u00cc\3\2\2\2\u00ce\u00cf\3\2\2\2\u00cf\u00d0\3\2"+
		"\2\2\u00d0\u00d1\5*\26\2\u00d1\u00d4\3\2\2\2\u00d2\u00d4\5\20\t\2\u00d3"+
		"\u00c4\3\2\2\2\u00d3\u00d2\3\2\2\2\u00d4\17\3\2\2\2\u00d5\u00d6\7\24\2"+
		"\2\u00d6\u00e1\5(\25\2\u00d7\u00d8\7\t\2\2\u00d8\u00dd\7-\2\2\u00d9\u00da"+
		"\7\3\2\2\u00da\u00dc\7-\2\2\u00db\u00d9\3\2\2\2\u00dc\u00df\3\2\2\2\u00dd"+
		"\u00db\3\2\2\2\u00dd\u00de\3\2\2\2\u00de\u00e0\3\2\2\2\u00df\u00dd\3\2"+
		"\2\2\u00e0\u00e2\7\n\2\2\u00e1\u00d7\3\2\2\2\u00e1\u00e2\3\2\2\2\u00e2"+
		"\u00e3\3\2\2\2\u00e3\u00e5\7\16\2\2\u00e4\u00e6\5\22\n\2\u00e5\u00e4\3"+
		"\2\2\2\u00e6\u00e7\3\2\2\2\u00e7\u00e5\3\2\2\2\u00e7\u00e8\3\2\2\2\u00e8"+
		"\21\3\2\2\2\u00e9\u00ea\7\25\2\2\u00ea\u00f6\5(\25\2\u00eb\u00ec\7\4\2"+
		"\2\u00ec\u00f1\5 \21\2\u00ed\u00ee\7\26\2\2\u00ee\u00f0\5 \21\2\u00ef"+
		"\u00ed\3\2\2\2\u00f0\u00f3\3\2\2\2\u00f1\u00ef\3\2\2\2\u00f1\u00f2\3\2"+
		"\2\2\u00f2\u00f4\3\2\2\2\u00f3\u00f1\3\2\2\2\u00f4\u00f5\7\5\2\2\u00f5"+
		"\u00f7\3\2\2\2\u00f6\u00eb\3\2\2\2\u00f6\u00f7\3\2\2\2\u00f7\23\3\2\2"+
		"\2\u00f8\u0103\5\26\f\2\u00f9\u00fa\5\30\r\2\u00fa\u00fb\7\3\2\2\u00fb"+
		"\u00fd\3\2\2\2\u00fc\u00f9\3\2\2\2\u00fd\u0100\3\2\2\2\u00fe\u00fc\3\2"+
		"\2\2\u00fe\u00ff\3\2\2\2\u00ff\u0101\3\2\2\2\u0100\u00fe\3\2\2\2\u0101"+
		"\u0103\5\32\16\2\u0102\u00f8\3\2\2\2\u0102\u00fe\3\2\2\2\u0103\25\3\2"+
		"\2\2\u0104\u0109\5\30\r\2\u0105\u0106\7\3\2\2\u0106\u0108\5\30\r\2\u0107"+
		"\u0105\3\2\2\2\u0108\u010b\3\2\2\2\u0109\u0107\3\2\2\2\u0109\u010a\3\2"+
		"\2\2\u010a\u010d\3\2\2\2\u010b\u0109\3\2\2\2\u010c\u0104\3\2\2\2\u010c"+
		"\u010d\3\2\2\2\u010d\27\3\2\2\2\u010e\u0111\7-\2\2\u010f\u0110\7\27\2"+
		"\2\u0110\u0112\5 \21\2\u0111\u010f\3\2\2\2\u0111\u0112\3\2\2\2\u0112\31"+
		"\3\2\2\2\u0113\u0118\5\34\17\2\u0114\u0115\7\3\2\2\u0115\u0117\5\34\17"+
		"\2\u0116\u0114\3\2\2\2\u0117\u011a\3\2\2\2\u0118\u0116\3\2\2\2\u0118\u0119"+
		"\3\2\2\2\u0119\33\3\2\2\2\u011a\u0118\3\2\2\2\u011b\u011c\7+\2\2\u011c"+
		"\u011d\7\16\2\2\u011d\u011e\5\n\6\2\u011e\35\3\2\2\2\u011f\u0120\7\t\2"+
		"\2\u0120\u012d\7\n\2\2\u0121\u0122\7\t\2\2\u0122\u0127\5.\30\2\u0123\u0124"+
		"\7\3\2\2\u0124\u0126\5.\30\2\u0125\u0123\3\2\2\2\u0126\u0129\3\2\2\2\u0127"+
		"\u0125\3\2\2\2\u0127\u0128\3\2\2\2\u0128\u012a\3\2\2\2\u0129\u0127\3\2"+
		"\2\2\u012a\u012b\7\n\2\2\u012b\u012d\3\2\2\2\u012c\u011f\3\2\2\2\u012c"+
		"\u0121\3\2\2\2\u012d\37\3\2\2\2\u012e\u012f\7\4\2\2\u012f\u015c\7\5\2"+
		"\2\u0130\u0131\7\4\2\2\u0131\u0132\5 \21\2\u0132\u0133\7\3\2\2\u0133\u0134"+
		"\7\5\2\2\u0134\u015c\3\2\2\2\u0135\u0136\7\4\2\2\u0136\u0139\5 \21\2\u0137"+
		"\u0138\7\3\2\2\u0138\u013a\5 \21\2\u0139\u0137\3\2\2\2\u013a\u013b\3\2"+
		"\2\2\u013b\u0139\3\2\2\2\u013b\u013c\3\2\2\2\u013c\u013d\3\2\2\2\u013d"+
		"\u013e\7\5\2\2\u013e\u015c\3\2\2\2\u013f\u015c\5(\25\2\u0140\u0141\7\30"+
		"\2\2\u0141\u0142\7\t\2\2\u0142\u0143\5\"\22\2\u0143\u0144\7\3\2\2\u0144"+
		"\u0145\5 \21\2\u0145\u0146\7\n\2\2\u0146\u015c\3\2\2\2\u0147\u0149\7\21"+
		"\2\2\u0148\u014a\5\36\20\2\u0149\u0148\3\2\2\2\u0149\u014a\3\2\2\2\u014a"+
		"\u014b\3\2\2\2\u014b\u0154\7\4\2\2\u014c\u0151\5 \21\2\u014d\u014e\7\3"+
		"\2\2\u014e\u0150\5 \21\2\u014f\u014d\3\2\2\2\u0150\u0153\3\2\2\2\u0151"+
		"\u014f\3\2\2\2\u0151\u0152\3\2\2\2\u0152\u0155\3\2\2\2\u0153\u0151\3\2"+
		"\2\2\u0154\u014c\3\2\2\2\u0154\u0155\3\2\2\2\u0155\u0156\3\2\2\2\u0156"+
		"\u0157\7\5\2\2\u0157\u0158\7\22\2\2\u0158\u015c\5 \21\2\u0159\u015c\7"+
		"\31\2\2\u015a\u015c\7\61\2\2\u015b\u012e\3\2\2\2\u015b\u0130\3\2\2\2\u015b"+
		"\u0135\3\2\2\2\u015b\u013f\3\2\2\2\u015b\u0140\3\2\2\2\u015b\u0147\3\2"+
		"\2\2\u015b\u0159\3\2\2\2\u015b\u015a\3\2\2\2\u015c!\3\2\2\2\u015d\u015e"+
		"\7\4\2\2\u015e\u0161\5&\24\2\u015f\u0160\7\3\2\2\u0160\u0162\5&\24\2\u0161"+
		"\u015f\3\2\2\2\u0162\u0163\3\2\2\2\u0163\u0161\3\2\2\2\u0163\u0164\3\2"+
		"\2\2\u0164\u0165\3\2\2\2\u0165\u0166\7\5\2\2\u0166\u016b\3\2\2\2\u0167"+
		"\u0168\7\4\2\2\u0168\u016b\7\5\2\2\u0169\u016b\5&\24\2\u016a\u015d\3\2"+
		"\2\2\u016a\u0167\3\2\2\2\u016a\u0169\3\2\2\2\u016b#\3\2\2\2\u016c\u016d"+
		"\7\32\2\2\u016d\u016e\7\t\2\2\u016e\u016f\7+\2\2\u016f\u0170\7\n\2\2\u0170"+
		"\u0171\7\t\2\2\u0171\u0172\7\61\2\2\u0172\u0173\7\n\2\2\u0173%\3\2\2\2"+
		"\u0174\u017b\5$\23\2\u0175\u0176\7\4\2\2\u0176\u0177\5&\24\2\u0177\u0178"+
		"\7\5\2\2\u0178\u017b\3\2\2\2\u0179\u017b\7\61\2\2\u017a\u0174\3\2\2\2"+
		"\u017a\u0175\3\2\2\2\u017a\u0179\3\2\2\2\u017b\'\3\2\2\2\u017c\u017d\7"+
		"+\2\2\u017d)\3\2\2\2\u017e\u017f\7\6\2\2\u017f\u0180\5\n\6\2\u0180\u0181"+
		"\7\7\2\2\u0181+\3\2\2\2\u0182\u0186\7\60\2\2\u0183\u0186\7\61\2\2\u0184"+
		"\u0186\7*\2\2\u0185\u0182\3\2\2\2\u0185\u0183\3\2\2\2\u0185\u0184\3\2"+
		"\2\2\u0186-\3\2\2\2\u0187\u018c\5\2\2\2\u0188\u018c\7,\2\2\u0189\u018c"+
		"\7-\2\2\u018a\u018c\7.\2\2\u018b\u0187\3\2\2\2\u018b\u0188\3\2\2\2\u018b"+
		"\u0189\3\2\2\2\u018b\u018a\3\2\2\2\u018c/\3\2\2\2+\66:=FIQUpz}\u0099\u00b2"+
		"\u00b4\u00b9\u00c0\u00c7\u00ce\u00d3\u00dd\u00e1\u00e7\u00f1\u00f6\u00fe"+
		"\u0102\u0109\u010c\u0111\u0118\u0127\u012c\u013b\u0149\u0151\u0154\u015b"+
		"\u0163\u016a\u017a\u0185\u018b";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}