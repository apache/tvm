/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api.h
 * \brief C API of NNVM symbolic construction and pass.
 *  Enables construction and transformation of Graph
 *  in any other host languages.
 */
#ifndef NNVM_C_API_H_
#define NNVM_C_API_H_

#ifdef __cplusplus
#define NNVM_EXTERN_C extern "C"
#endif

/*! \brief NNVM_DLL prefix for windows */
#ifdef _WIN32
#ifdef NNVM_EXPORTS
#define NNVM_DLL NNVM_EXTERN_C __declspec(dllexport)
#else
#define NNVM_DLL NNVM_EXTERN_C __declspec(dllimport)
#endif
#else
#define NNVM_DLL NNVM_EXTERN_C
#endif

/*! \brief manually define unsigned int */
typedef unsigned int nn_uint;

/*! \brief handle to a function that takes param and creates symbol */
typedef void *AtomicSymbolCreator;
/*! \brief handle to a symbol that can be bind as operator */
typedef void *SymbolHandle;
/*! \brief handle to Graph */
typedef void *GraphHandle;

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  NNGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
NNVM_DLL const char *NNGetLastError();

/*!
 * \brief list all the available AtomicSymbolEntry
 * \param out_size the size of returned array
 * \param out_array the output AtomicSymbolCreator array
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolListAtomicSymbolCreators(nn_uint *out_size,
                                              AtomicSymbolCreator **out_array);
/*!
 * \brief Get the detailed information about atomic symbol.
 * \param creator the AtomicSymbolCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_doc_args Number of arguments that contain documents.
 * \param arg_names Name of the arguments of doc args
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param return_type Return type of the function, if any.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                         const char **name,
                                         const char **description,
                                         nn_uint *num_doc_args,
                                         const char ***arg_names,
                                         const char ***arg_type_infos,
                                         const char ***arg_descriptions,
                                         const char **return_type);
/*!
 * \brief Create an AtomicSymbol functor.
 * \param creator the AtomicSymbolCreator
 * \param num_param the number of parameters
 * \param keys the keys to the params
 * \param vals the vals of the params
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                                        nn_uint num_param,
                                        const char **keys,
                                        const char **vals,
                                        SymbolHandle *out);
/*!
 * \brief Create a Variable Symbol.
 * \param name name of the variable
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolCreateVariable(const char *name, SymbolHandle *out);
/*!
 * \brief Create a Symbol by grouping list of symbols together
 * \param num_symbols number of symbols to be grouped
 * \param symbols array of symbol handles
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolCreateGroup(nn_uint num_symbols,
                                 SymbolHandle *symbols,
                                 SymbolHandle *out);
/*!
 * \brief Free the symbol handle.
 * \param symbol the symbol
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolFree(SymbolHandle symbol);
/*!
 * \brief Copy the symbol to another handle
 * \param symbol the source symbol
 * \param out used to hold the result of copy
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolCopy(SymbolHandle symbol, SymbolHandle *out);
/*!
 * \brief Print the content of symbol, used for debug.
 * \param symbol the symbol
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolPrint(SymbolHandle symbol, const char **out_str);
/*!
 * \brief Get string attribute from symbol
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolGetAttr(SymbolHandle symbol,
                             const char* key,
                             const char** out,
                             int *success);
/*!
 * \brief Set string attribute from symbol.
 *  NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
 *
 *  Safe recommendaton: use  immutable graph
 *  - Only allow set attributes during creation of new symbol as optional parameter
 *
 *  Mutable graph (be careful about the semantics):
 *  - Allow set attr at any point.
 *  - Mutating an attribute of some common node of two graphs can cause confusion from user.
 *
 * \param symbol the source symbol
 * \param num_param Number of parameters to set.
 * \param keys The keys of the attribute
 * \param values The value to be set
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolSetAttrs(SymbolHandle symbol,
                              nn_uint num_param,
                              const char** keys,
                              const char** values);
/*!
 * \brief Get all attributes from symbol, including all descendents.
 * \param symbol the source symbol
 * \param recursive_option 0 for recursive, 1 for shallow.
 * \param out_size The number of output attributes
 * \param out 2*out_size strings representing key value pairs.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolListAttrs(SymbolHandle symbol,
                               int recursive_option,
                               nn_uint *out_size,
                               const char*** out);
/*!
 * \brief List arguments in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolListArguments(SymbolHandle symbol,
                                   nn_uint *out_size,
                                   const char ***out_str_array);
/*!
 * \brief List returns in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolListOutputs(SymbolHandle symbol,
                                 nn_uint *out_size,
                                 const char ***out_str_array);
/*!
 * \brief Get a symbol that contains all the internals.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are all the internals.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolGetInternals(SymbolHandle symbol,
                                  SymbolHandle *out);
/*!
 * \brief Get index-th outputs of the symbol.
 * \param symbol The symbol
 * \param index the Index of the output.
 * \param out The output symbol whose outputs are the index-th symbol.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolGetOutput(SymbolHandle symbol,
                               nn_uint index,
                               SymbolHandle *out);

/*!
 * \brief Compose the symbol on other symbols.
 *
 *  This function will change the sym hanlde.
 *  To achieve function apply behavior, copy the symbol first
 *  before apply.
 *
 * \param sym the symbol to apply
 * \param name the name of symbol
 * \param num_args number of arguments
 * \param keys the key of keyword args (optional)
 * \param args arguments to sym
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNSymbolCompose(SymbolHandle sym,
                             const char* name,
                             nn_uint num_args,
                             const char** keys,
                             SymbolHandle* args);

// Graph IR API
/*!
 * \brief create a graph handle from symbol
 * \param symbol The symbol representing the graph.
 * \param graph The graph handle created.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNGraphCreate(SymbolHandle symbol, GraphHandle *graph);
/*!
 * \brief free the graph handle
 * \param handle The handle to be freed.
 */
NNVM_DLL int NNGraphFree(GraphHandle handle);
/*!
 * \brief Get a new symbol from the graph.
 * \param graph The graph handle.
 * \param symbol The corresponding symbol
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNGraphGetSymbol(GraphHandle graph, SymbolHandle *symbol);
/*!
 * \brief Get Set a std::string typed attribute to graph.
 * \param handle The graph handle.
 * \param key The key to the attribute.
 * \param value The value to be exposed.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNGraphSetStrAttr(GraphHandle handle,
                               const char* key,
                               const char* value);
/*!
 * \brief Get Set a std::string typed attribute from graph attribute.
 * \param handle The graph handle.
 * \param key The key to the attribute.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNGraphGetStrAttr(SymbolHandle handle,
                               const char* key,
                               const char** out,
                               int *success);
/*!
 * \brief Apply pass on the src graph.
 * \param src The source graph handle.
 * \param num_pass The number of pass to be applied.
 * \param pass_names The names of the pass.
 * \param dst The result graph.
 * \return 0 when success, -1 when failure happens
 */
NNVM_DLL int NNGraphApplyPass(GraphHandle src,
                              nn_uint num_pass,
                              const char** pass_names,
                              GraphHandle *dst);

#endif  // NNVM_C_API_H_
