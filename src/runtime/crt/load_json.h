/*!
 * Copyright (c) 2015 by Contributors
 * \file json.h
 * \brief Lightweight JSON Reader that read save into C++ data structs.
 *  This includes STL composites and structures.
 */
#ifndef LOAD_JSON_H_
#define LOAD_JSON_H_

#include "common.h"

enum {
  JSON_READ_TYPE_U8 = 1,
  JSON_READ_TYPE_S8 = 2,
  JSON_READ_TYPE_U16 = 3,
  JSON_READ_TYPE_S16 = 4,
  JSON_READ_TYPE_U32 = 5,
  JSON_READ_TYPE_S32 = 6,
  JSON_READ_TYPE_F32 = 7,
  JSON_READ_TYPE_F64 = 8,
  JSON_READ_TYPE_GRAPH_RUNTIME_NODE = 9,
  JSON_READ_TYPE_GRAPH_RUNTIME_NODE_ENTRY = 10,
  JSON_READ_TYPE_GRAPH_RUNTIME_GRAPH_ATTR = 11
};

/*!
 * \brief Lightweight JSON Reader to read any STL compositions and structs.
 *  The user need to know the schema of the
 */
typedef struct json_reader_t {
  /*! \brief internal reader string */
  char is_[TVM_CRT_MAX_JSON_LENGTH];
  char * isptr;
  /*! \brief "\\r" counter */
  size_t line_count_r_;
  /*! \brief "\\n" counter */
  size_t line_count_n_;
  /*!
   * \brief record how many element processed in
   *  current array/object scope.
   */
  Seq scope_counter_;

  char (*NextChar)(struct json_reader_t * reader);
  char (*NextNonSpace)(struct json_reader_t * reader);
  char (*PeekNextChar)(struct json_reader_t * reader);
  char (*PeekNextNonSpace)(struct json_reader_t * reader);
  int (*ReadUnsignedInteger)(struct json_reader_t * reader, unsigned int * out_value);
  int (*ReadInteger)(struct json_reader_t * reader, int64_t * out_value);
  int (*ReadString)(struct json_reader_t * reader, char * out_value);
  void (*BeginArray)(struct json_reader_t * reader);
  void (*BeginObject)(struct json_reader_t * reader);
  bool (*NextObjectItem)(struct json_reader_t * reader, char * out_key);
  bool (*NextArrayItem)(struct json_reader_t * reader);
} JSONReader;

typedef void (*ReadFunction)(JSONReader *reader, void *addr);

/*! \brief internal data entry */
struct JSONObjectReadHelperEntry {
  /*! \brief the reader function */
  ReadFunction func;
  /*! \brief the address to read */
  void *addr;
  /*! \brief whether it is optional */
  bool optional;
};

/*!
 * \brief Helper class to read JSON into a class or struct object.
 * \code
 *  struct Param {
 *    string name;
 *    int value;
 *    // define load function from JSON
 *    inline void Load(dmlc::JSONReader *reader) {
 *      dmlc::JSONStructReadHelper helper;
 *      helper.DeclareField("name", &name);
 *      helper.DeclareField("value", &value);
 *      helper.ReadAllFields(reader);
 *    }
 *  };
 * \endcode
 */
struct JSONObjectReadHelper {
  /*!
   * \brief Read in all the declared fields.
   * \param reader the JSONReader to read the json.
   */
  void (*ReadAllFields)(JSONReader *reader);
  /*!
   * \brief The internal reader function.
   * \param reader The reader to read.
   * \param addr The memory address to read.
   */
  void (*ReaderFunction)(JSONReader *reader, void *addr);
};

#define DMLC_JSON_ENABLE_ANY_VAR_DEF(KeyName)                  \
  static DMLC_ATTRIBUTE_UNUSED ::dmlc::json::AnyJSONManager&   \
  __make_AnyJSONType ## _ ## KeyName ## __

/*!
 * \def DMLC_JSON_ENABLE_ANY
 * \brief Macro to enable save/load JSON of dmlc:: whose actual type is Type.
 * Any type will be saved as json array [KeyName, content]
 *
 * \param Type The type to be registered.
 * \param KeyName The Type key assigned to the type, must be same during load.
 */
#define DMLC_JSON_ENABLE_ANY(Type, KeyName)                             \
  DMLC_STR_CONCAT(DMLC_JSON_ENABLE_ANY_VAR_DEF(KeyName), __COUNTER__) = \
    ::dmlc::json::AnyJSONManager::Global()->EnableType<Type>(#KeyName) \

// implementations of JSONReader

/*!
 * \brief Takes the next char from the input source.
 * \return the next character.
 */
static inline char JSONReader_NextChar(JSONReader * reader) {
  char ch = reader->isptr[0];
  reader->isptr += 1;
  return ch;
}

/*!
 * \brief Returns the next char from the input source.
 * \return the next character.
 */
static inline char JSONReader_PeekNextChar(JSONReader * reader) {
  return reader->isptr[0];
}

/*!
 * \brief Read next nonspace character.
 * \return the next nonspace character.
 */
static inline char JSONReader_NextNonSpace(JSONReader * reader) {
  int ch;
  do {
    ch = reader->NextChar(reader);
    if (ch == '\n') { ++(reader->line_count_n_); }
    if (ch == '\r') { ++(reader->line_count_r_); }
  } while (isspace(ch));
  return ch;
}

/*!
 * \brief Read just before next nonspace but not read that.
 * \return the next nonspace character.
 */
static inline char JSONReader_PeekNextNonSpace(JSONReader * reader) {
  int ch;
  while (true) {
    ch = reader->PeekNextChar(reader);
    if (ch == '\n') { ++(reader->line_count_n_); }
    if (ch == '\r') { ++(reader->line_count_r_); }
    if (!isspace(ch)) break;
    reader->NextChar(reader);
  }
  return ch;
}

/*!
 * \brief Parse next JSON string.
 * \param out_str the output string.
 * \throw dmlc::Error when next token is not string
 */
static inline int JSONReader_ReadString(JSONReader * reader, char * out_str) {
  int status = TVM_STATUS_SUCCESS;
  char ch = reader->NextNonSpace(reader);
  char output[128];
  uint32_t output_counter = 0;
  memset(output, 0, 128);
  while (true) {
    ch = reader->NextChar(reader);
    if (ch == '\\') {
      char sch = reader->NextChar(reader);
      switch (sch) {
        case 'r': strcat(output, "\r"); break;
        case 'n': strcat(output, "\n"); break;
        case '\\': strcat(output, "\\"); break;
        case 't': strcat(output, "\t"); break;
        case '\"': strcat(output, "\""); break;
        default: LOGE("unknown string escape \%c", sch);
      }
    } else {
      if (ch == '\"') { break; }
      if (strlen(output) >= 127) {
        LOGE("Error: detected buffer overflow.");
        status = TVM_STATUS_FAILURE;
        break;
      }
      strncat(output, &ch, 1);
      output_counter++;
      if (output_counter >= 127) {
        LOGE("Error: string size greater than 128.");
        status = TVM_STATUS_FAILURE;
        break;
      }
    }
    if (ch == EOF || ch == '\r' || ch == '\n') {
      LOGE("Error at line X, Expect \'\"\' but reach end of line");
    }
  }
  strcpy(out_str, output);
  return status;
}

static inline int JSONReader_ReadUnsignedInteger(JSONReader * reader, unsigned int * out_value) {
  int status = TVM_STATUS_SUCCESS;
  char* endptr;
  const char* icstr = reader->isptr; // ->data_;
  unsigned int number = strtol(icstr, &endptr, 10);
  reader->isptr += endptr - icstr;
  *out_value = number;
  return status;
}


static inline int JSONReader_ReadInteger(JSONReader * reader, int64_t * out_value) {
  int status = TVM_STATUS_SUCCESS;
  char* endptr;
  const char* icstr = reader->isptr; // ->data_;
  int64_t number = strtol(icstr, &endptr, 10);
  reader->isptr += endptr - icstr;
  *out_value = number;
  return status;
}

/*!
 * \brief Begin parsing an object.
 * \code
 *  string key;
 *  // value can be any type that is json serializable.
 *  string value;
 *  reader->BeginObject();
 *  while (reader->NextObjectItem(&key)) {
 *    // do somthing to key value
 *    reader->Read(&value);
 *  }
 * \endcode
 */
static inline void JSONReader_BeginObject(JSONReader * reader) {
  int ch = reader->NextNonSpace(reader);
  if (!(ch == '{')) {
    LOGE("Error at line X, Expect \'{\' but got \'%c\'", ch);
  }
  Seq * scope_counter_ = &(reader->scope_counter_);
  scope_counter_->push_back(scope_counter_, 0);
}

/*!
 * \brief Try to move to next object item.
 *  If this call is successful, user can proceed to call
 *  reader->Read to read in the value.
 * \param out_key the key to the next object.
 * \return true if the read is successful, false if we are at end of the object.
 */
static inline bool JSONReader_NextObjectItem(JSONReader * reader, char * out_key) {
  bool next = true;
  Seq * scope_counter_ = &(reader->scope_counter_);
  if (scope_counter_->back(scope_counter_)[0] != 0) {
    int ch = reader->NextNonSpace(reader);
    if (ch == EOF) {
      next = false;
    } else if (ch == '}') {
      next = false;
    } else {
      if (ch != ',') {
        LOGE("Error at line X, JSON object expect \'}\' or \',\' but got \'%c\'", ch);
      }
    }
  } else {
    int ch = reader->PeekNextNonSpace(reader);
    if (ch == '}') {
      reader->NextChar(reader);
      next = false;
    }
  }
  if (!next) {
    scope_counter_->pop_back(scope_counter_);
    return false;
  } else {
    scope_counter_->back(scope_counter_)[0] += 1;
    reader->ReadString(reader, out_key);
    int ch = reader->NextNonSpace(reader);
    if (ch != ':') {
      LOGE("Error at line X, Expect \':\' but get \'%c\'", ch);
    }
    return true;
  }
}

/*!
 * \brief Begin parsing an array.
 * \code
 *  // value can be any type that is json serializable.
 *  string value;
 *  reader->BeginArray();
 *  while (reader->NextArrayItem(&value)) {
 *    // do somthing to value
 *  }
 * \endcode
 */
static inline void JSONReader_BeginArray(JSONReader * reader) {
  int ch = reader->NextNonSpace(reader);
  if (ch != '[') {
    LOGE("Error at line X, Expect \'[\' but get \'%c\'", ch);
  }
  Seq * scope_counter_ = &(reader->scope_counter_);
  scope_counter_->push_back(scope_counter_, 0);
}

/*!
 * \brief Try to read the next element in the array.
 *  If this call is successful, user can proceed to call
 *  reader->Read to read in the value.
 * \return true if the read is successful, false if we are at end of the array.
 */
static inline bool JSONReader_NextArrayItem(JSONReader * reader) {
  bool next = true;
  Seq * scope_counter_ = &(reader->scope_counter_);
  if (scope_counter_->back(scope_counter_)[0] != 0) {
    int ch = reader->NextNonSpace(reader);
    if (ch == EOF) {
      next = false;
    } else if (ch == ']') {
      next = false;
    } else {
      if (ch != ',') {
        LOGE("Error at line X, JSON object expect \']\' or \',\' but got \'%c\'", ch);
      }
    }
  } else {
    int ch = reader->PeekNextNonSpace(reader);
    if (ch == ']') {
      reader->NextChar(reader);
      next = false;
    }
  }
  if (!next) {
    scope_counter_->pop_back(scope_counter_);
    return false;
  } else {
    scope_counter_->back(scope_counter_)[0] += 1;
    return true;
  }
}

/*!
 * \brief Constructor.
 * \param is the input source.
 */
static inline JSONReader JSONReader_Create(const char * is) {
  JSONReader reader; // = (JSONReader*)malloc(sizeof(JSONReader));
  memset(&reader, 0, sizeof(JSONReader));
  reader.scope_counter_ = SeqCreate();
  reader.NextChar = JSONReader_NextChar;
  reader.PeekNextChar = JSONReader_PeekNextChar;
  reader.NextNonSpace = JSONReader_NextNonSpace;
  reader.PeekNextNonSpace = JSONReader_PeekNextNonSpace;
  reader.ReadString = JSONReader_ReadString;
  reader.ReadUnsignedInteger = JSONReader_ReadUnsignedInteger;
  reader.ReadInteger = JSONReader_ReadInteger;
  reader.BeginArray = JSONReader_BeginArray;
  reader.BeginObject = JSONReader_BeginObject;
  reader.NextArrayItem = JSONReader_NextArrayItem;
  reader.NextObjectItem = JSONReader_NextObjectItem;
  strcpy(reader.is_, is);
  reader.isptr = reader.is_;
  return reader;
}

#endif  // LOAD_JSON_H_
