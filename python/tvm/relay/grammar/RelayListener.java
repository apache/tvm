// Generated from Relay.g4 by ANTLR 4.7.2
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link RelayParser}.
 */
public interface RelayListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link RelayParser#opIdent}.
	 * @param ctx the parse tree
	 */
	void enterOpIdent(RelayParser.OpIdentContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#opIdent}.
	 * @param ctx the parse tree
	 */
	void exitOpIdent(RelayParser.OpIdentContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#prog}.
	 * @param ctx the parse tree
	 */
	void enterProg(RelayParser.ProgContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#prog}.
	 * @param ctx the parse tree
	 */
	void exitProg(RelayParser.ProgContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#exprList}.
	 * @param ctx the parse tree
	 */
	void enterExprList(RelayParser.ExprListContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#exprList}.
	 * @param ctx the parse tree
	 */
	void exitExprList(RelayParser.ExprListContext ctx);
	/**
	 * Enter a parse tree produced by the {@code callNoAttr}
	 * labeled alternative in {@link RelayParser#callList}.
	 * @param ctx the parse tree
	 */
	void enterCallNoAttr(RelayParser.CallNoAttrContext ctx);
	/**
	 * Exit a parse tree produced by the {@code callNoAttr}
	 * labeled alternative in {@link RelayParser#callList}.
	 * @param ctx the parse tree
	 */
	void exitCallNoAttr(RelayParser.CallNoAttrContext ctx);
	/**
	 * Enter a parse tree produced by the {@code callWithAttr}
	 * labeled alternative in {@link RelayParser#callList}.
	 * @param ctx the parse tree
	 */
	void enterCallWithAttr(RelayParser.CallWithAttrContext ctx);
	/**
	 * Exit a parse tree produced by the {@code callWithAttr}
	 * labeled alternative in {@link RelayParser#callList}.
	 * @param ctx the parse tree
	 */
	void exitCallWithAttr(RelayParser.CallWithAttrContext ctx);
	/**
	 * Enter a parse tree produced by the {@code funcExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterFuncExpr(RelayParser.FuncExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code funcExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitFuncExpr(RelayParser.FuncExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code metaExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterMetaExpr(RelayParser.MetaExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code metaExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitMetaExpr(RelayParser.MetaExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code tensor}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterTensor(RelayParser.TensorContext ctx);
	/**
	 * Exit a parse tree produced by the {@code tensor}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitTensor(RelayParser.TensorContext ctx);
	/**
	 * Enter a parse tree produced by the {@code graph}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterGraph(RelayParser.GraphContext ctx);
	/**
	 * Exit a parse tree produced by the {@code graph}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitGraph(RelayParser.GraphContext ctx);
	/**
	 * Enter a parse tree produced by the {@code identExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterIdentExpr(RelayParser.IdentExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code identExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitIdentExpr(RelayParser.IdentExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code stringExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterStringExpr(RelayParser.StringExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code stringExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitStringExpr(RelayParser.StringExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code call}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterCall(RelayParser.CallContext ctx);
	/**
	 * Exit a parse tree produced by the {@code call}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitCall(RelayParser.CallContext ctx);
	/**
	 * Enter a parse tree produced by the {@code neg}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterNeg(RelayParser.NegContext ctx);
	/**
	 * Exit a parse tree produced by the {@code neg}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitNeg(RelayParser.NegContext ctx);
	/**
	 * Enter a parse tree produced by the {@code tuple}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterTuple(RelayParser.TupleContext ctx);
	/**
	 * Exit a parse tree produced by the {@code tuple}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitTuple(RelayParser.TupleContext ctx);
	/**
	 * Enter a parse tree produced by the {@code paren}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterParen(RelayParser.ParenContext ctx);
	/**
	 * Exit a parse tree produced by the {@code paren}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitParen(RelayParser.ParenContext ctx);
	/**
	 * Enter a parse tree produced by the {@code scalarExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterScalarExpr(RelayParser.ScalarExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code scalarExpr}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitScalarExpr(RelayParser.ScalarExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code let}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterLet(RelayParser.LetContext ctx);
	/**
	 * Exit a parse tree produced by the {@code let}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitLet(RelayParser.LetContext ctx);
	/**
	 * Enter a parse tree produced by the {@code projection}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterProjection(RelayParser.ProjectionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code projection}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitProjection(RelayParser.ProjectionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ifElse}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterIfElse(RelayParser.IfElseContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ifElse}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitIfElse(RelayParser.IfElseContext ctx);
	/**
	 * Enter a parse tree produced by the {@code binOp}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterBinOp(RelayParser.BinOpContext ctx);
	/**
	 * Exit a parse tree produced by the {@code binOp}
	 * labeled alternative in {@link RelayParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitBinOp(RelayParser.BinOpContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#func}.
	 * @param ctx the parse tree
	 */
	void enterFunc(RelayParser.FuncContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#func}.
	 * @param ctx the parse tree
	 */
	void exitFunc(RelayParser.FuncContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#defn}.
	 * @param ctx the parse tree
	 */
	void enterDefn(RelayParser.DefnContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#defn}.
	 * @param ctx the parse tree
	 */
	void exitDefn(RelayParser.DefnContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#adtDefn}.
	 * @param ctx the parse tree
	 */
	void enterAdtDefn(RelayParser.AdtDefnContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#adtDefn}.
	 * @param ctx the parse tree
	 */
	void exitAdtDefn(RelayParser.AdtDefnContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#adtVariant}.
	 * @param ctx the parse tree
	 */
	void enterAdtVariant(RelayParser.AdtVariantContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#adtVariant}.
	 * @param ctx the parse tree
	 */
	void exitAdtVariant(RelayParser.AdtVariantContext ctx);
	/**
	 * Enter a parse tree produced by the {@code argNoAttr}
	 * labeled alternative in {@link RelayParser#argList}.
	 * @param ctx the parse tree
	 */
	void enterArgNoAttr(RelayParser.ArgNoAttrContext ctx);
	/**
	 * Exit a parse tree produced by the {@code argNoAttr}
	 * labeled alternative in {@link RelayParser#argList}.
	 * @param ctx the parse tree
	 */
	void exitArgNoAttr(RelayParser.ArgNoAttrContext ctx);
	/**
	 * Enter a parse tree produced by the {@code argWithAttr}
	 * labeled alternative in {@link RelayParser#argList}.
	 * @param ctx the parse tree
	 */
	void enterArgWithAttr(RelayParser.ArgWithAttrContext ctx);
	/**
	 * Exit a parse tree produced by the {@code argWithAttr}
	 * labeled alternative in {@link RelayParser#argList}.
	 * @param ctx the parse tree
	 */
	void exitArgWithAttr(RelayParser.ArgWithAttrContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#varList}.
	 * @param ctx the parse tree
	 */
	void enterVarList(RelayParser.VarListContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#varList}.
	 * @param ctx the parse tree
	 */
	void exitVarList(RelayParser.VarListContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#var}.
	 * @param ctx the parse tree
	 */
	void enterVar(RelayParser.VarContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#var}.
	 * @param ctx the parse tree
	 */
	void exitVar(RelayParser.VarContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#attrSeq}.
	 * @param ctx the parse tree
	 */
	void enterAttrSeq(RelayParser.AttrSeqContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#attrSeq}.
	 * @param ctx the parse tree
	 */
	void exitAttrSeq(RelayParser.AttrSeqContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#attr}.
	 * @param ctx the parse tree
	 */
	void enterAttr(RelayParser.AttrContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#attr}.
	 * @param ctx the parse tree
	 */
	void exitAttr(RelayParser.AttrContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#typeParamList}.
	 * @param ctx the parse tree
	 */
	void enterTypeParamList(RelayParser.TypeParamListContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#typeParamList}.
	 * @param ctx the parse tree
	 */
	void exitTypeParamList(RelayParser.TypeParamListContext ctx);
	/**
	 * Enter a parse tree produced by the {@code tupleType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void enterTupleType(RelayParser.TupleTypeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code tupleType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void exitTupleType(RelayParser.TupleTypeContext ctx);
	/**
	 * Enter a parse tree produced by the {@code typeIdentType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void enterTypeIdentType(RelayParser.TypeIdentTypeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code typeIdentType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void exitTypeIdentType(RelayParser.TypeIdentTypeContext ctx);
	/**
	 * Enter a parse tree produced by the {@code tensorType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void enterTensorType(RelayParser.TensorTypeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code tensorType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void exitTensorType(RelayParser.TensorTypeContext ctx);
	/**
	 * Enter a parse tree produced by the {@code funcType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void enterFuncType(RelayParser.FuncTypeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code funcType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void exitFuncType(RelayParser.FuncTypeContext ctx);
	/**
	 * Enter a parse tree produced by the {@code incompleteType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void enterIncompleteType(RelayParser.IncompleteTypeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code incompleteType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void exitIncompleteType(RelayParser.IncompleteTypeContext ctx);
	/**
	 * Enter a parse tree produced by the {@code intType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void enterIntType(RelayParser.IntTypeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code intType}
	 * labeled alternative in {@link RelayParser#type_}.
	 * @param ctx the parse tree
	 */
	void exitIntType(RelayParser.IntTypeContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#shapeList}.
	 * @param ctx the parse tree
	 */
	void enterShapeList(RelayParser.ShapeListContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#shapeList}.
	 * @param ctx the parse tree
	 */
	void exitShapeList(RelayParser.ShapeListContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#meta}.
	 * @param ctx the parse tree
	 */
	void enterMeta(RelayParser.MetaContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#meta}.
	 * @param ctx the parse tree
	 */
	void exitMeta(RelayParser.MetaContext ctx);
	/**
	 * Enter a parse tree produced by the {@code metaShape}
	 * labeled alternative in {@link RelayParser#shape}.
	 * @param ctx the parse tree
	 */
	void enterMetaShape(RelayParser.MetaShapeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code metaShape}
	 * labeled alternative in {@link RelayParser#shape}.
	 * @param ctx the parse tree
	 */
	void exitMetaShape(RelayParser.MetaShapeContext ctx);
	/**
	 * Enter a parse tree produced by the {@code parensShape}
	 * labeled alternative in {@link RelayParser#shape}.
	 * @param ctx the parse tree
	 */
	void enterParensShape(RelayParser.ParensShapeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code parensShape}
	 * labeled alternative in {@link RelayParser#shape}.
	 * @param ctx the parse tree
	 */
	void exitParensShape(RelayParser.ParensShapeContext ctx);
	/**
	 * Enter a parse tree produced by the {@code intShape}
	 * labeled alternative in {@link RelayParser#shape}.
	 * @param ctx the parse tree
	 */
	void enterIntShape(RelayParser.IntShapeContext ctx);
	/**
	 * Exit a parse tree produced by the {@code intShape}
	 * labeled alternative in {@link RelayParser#shape}.
	 * @param ctx the parse tree
	 */
	void exitIntShape(RelayParser.IntShapeContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#typeIdent}.
	 * @param ctx the parse tree
	 */
	void enterTypeIdent(RelayParser.TypeIdentContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#typeIdent}.
	 * @param ctx the parse tree
	 */
	void exitTypeIdent(RelayParser.TypeIdentContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#body}.
	 * @param ctx the parse tree
	 */
	void enterBody(RelayParser.BodyContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#body}.
	 * @param ctx the parse tree
	 */
	void exitBody(RelayParser.BodyContext ctx);
	/**
	 * Enter a parse tree produced by the {@code scalarFloat}
	 * labeled alternative in {@link RelayParser#scalar}.
	 * @param ctx the parse tree
	 */
	void enterScalarFloat(RelayParser.ScalarFloatContext ctx);
	/**
	 * Exit a parse tree produced by the {@code scalarFloat}
	 * labeled alternative in {@link RelayParser#scalar}.
	 * @param ctx the parse tree
	 */
	void exitScalarFloat(RelayParser.ScalarFloatContext ctx);
	/**
	 * Enter a parse tree produced by the {@code scalarInt}
	 * labeled alternative in {@link RelayParser#scalar}.
	 * @param ctx the parse tree
	 */
	void enterScalarInt(RelayParser.ScalarIntContext ctx);
	/**
	 * Exit a parse tree produced by the {@code scalarInt}
	 * labeled alternative in {@link RelayParser#scalar}.
	 * @param ctx the parse tree
	 */
	void exitScalarInt(RelayParser.ScalarIntContext ctx);
	/**
	 * Enter a parse tree produced by the {@code scalarBool}
	 * labeled alternative in {@link RelayParser#scalar}.
	 * @param ctx the parse tree
	 */
	void enterScalarBool(RelayParser.ScalarBoolContext ctx);
	/**
	 * Exit a parse tree produced by the {@code scalarBool}
	 * labeled alternative in {@link RelayParser#scalar}.
	 * @param ctx the parse tree
	 */
	void exitScalarBool(RelayParser.ScalarBoolContext ctx);
	/**
	 * Enter a parse tree produced by {@link RelayParser#ident}.
	 * @param ctx the parse tree
	 */
	void enterIdent(RelayParser.IdentContext ctx);
	/**
	 * Exit a parse tree produced by {@link RelayParser#ident}.
	 * @param ctx the parse tree
	 */
	void exitIdent(RelayParser.IdentContext ctx);
}