/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/ir/config_schema.h
 * \brief Minimal schema for dynamic config canonicalization and validation.
 *
 * This utility is intended for dynamic map-like configs (e.g. Target options),
 * where we still want type checking, optional defaulting, and canonicalization.
 */
#ifndef TVM_IR_CONFIG_SCHEMA_H_
#define TVM_IR_CONFIG_SCHEMA_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace ir {

/*!
 * \brief Dynamic config schema for map-like options.
 *
 * The schema supports:
 * - Option declaration (`def_option<T>`)
 * - Optional canonicalizer (`set_canonicalizer`)
 * - Resolution (`Resolve`) that performs validation/defaulting, unknown-key policy,
 *   and canonicalization (last).
 */
class ConfigSchema {
 public:
  using ConfigMap = ffi::Map<ffi::String, ffi::Any>;
  using Canonicalizer = ffi::TypedFunction<ConfigMap(ConfigMap)>;

  /*! \brief Schema entry for one declared option. */
  struct OptionEntry {
    /*! \brief Option key. */
    ffi::String key;
    /*! \brief Type string for this option. */
    ffi::String type_str;
    /*! \brief Per-option validator/coercer (Any -> Any). */
    ffi::TypedFunction<ffi::Any(ffi::Any)> validator;
    /*! \brief Whether this option has a default value. */
    bool has_default = false;
    /*! \brief Default value (valid when has_default is true). */
    ffi::Any default_value;
  };

  /*!
   * \brief Declare a typed option.
   *
   * Validation/coercion is implicitly generated from `T`.
   * Additional optional traits may be supplied (e.g. `refl::DefaultValue`, `const char*` doc).
   *
   * \tparam T The canonical value type of this option.
   * \tparam Traits Optional metadata/traits.
   * \param key Option key.
   * \param traits Optional traits.
   * \return Reference to `*this` for chaining.
   */
  template <typename T, typename... Traits>
  ConfigSchema& def_option(const ffi::String& key, Traits&&... traits) {
    std::string skey(key);
    if (key_to_index_.count(skey)) {
      TVM_FFI_THROW(ValueError) << "Duplicate config option key: '" << key << "'";
    }
    key_to_index_[skey] = options_.size();
    options_.push_back(MakeEntry<T>(key, std::forward<Traits>(traits)...));
    return *this;
  }

  /*! \brief Set whole-object canonicalizer. */
  void set_canonicalizer(Canonicalizer f) { canonicalizer_ = std::move(f); }

  /*! \brief Trait to set a custom validator for a config option. */
  struct AttrValidator {
    ffi::TypedFunction<ffi::Any(ffi::Any)> func;
    explicit AttrValidator(ffi::TypedFunction<ffi::Any(ffi::Any)> f) : func(std::move(f)) {}
  };

  /*! \brief Set whether unknown keys trigger an error. */
  void set_error_on_unknown(bool value) { error_on_unknown_ = value; }

  /*!
   * \brief Default/validate, then canonicalize a config object.
   *
   * Resolve flow:
   * 1) Validate/coerce declared options in declaration order.
   * 2) Materialize defaults and enforce required options.
   * 3) Apply unknown-key policy.
   * 4) Run canonicalizer as final step.
   *
   * \param config Input config object.
   * \return Canonical validated config object.
   * \throws ValueError/TypeError with option context.
   */
  ConfigMap Resolve(ConfigMap config) const {
    ConfigMap result;

    // Step 1: validate/coerce and materialize options in declaration order
    for (const auto& e : options_) {
      auto it = config.find(e.key);
      if (it != config.end()) {
        result.Set(e.key, e.validator((*it).second));
      } else if (e.has_default) {
        result.Set(e.key, e.default_value);
      }
      // else: missing non-required option, stays absent
    }

    // Step 2: unknown-key policy
    if (error_on_unknown_) {
      for (const auto& kv : config) {
        if (!key_to_index_.count(std::string(kv.first))) {
          std::ostringstream os;
          os << "Unknown config option '" << kv.first << "'. Known options: ";
          bool first = true;
          for (const auto& e : options_) {
            if (!first) os << ", ";
            os << "'" << e.key << "'";
            first = false;
          }
          TVM_FFI_THROW(ValueError) << os.str();
        }
      }
    }

    // Step 3: whole-object canonicalization (last)
    if (canonicalizer_ != nullptr) {
      result = canonicalizer_(result);
    }

    return result;
  }

  /*!
   * \brief List declared options in declaration order.
   * \return Const reference to the option entries vector.
   */
  const std::vector<OptionEntry>& ListOptions() const { return options_; }

  /*! \brief Check if an option with the given key exists. */
  bool HasOption(const ffi::String& key) const { return key_to_index_.count(std::string(key)) > 0; }

 private:
  template <typename T>
  static ffi::TypedFunction<ffi::Any(ffi::Any)> MakeValidator(const ffi::String& key) {
    return ffi::TypedFunction<ffi::Any(ffi::Any)>([key](ffi::Any val) -> ffi::Any {
      auto opt = val.try_cast<T>();
      if (!opt.has_value()) {
        TVM_FFI_THROW(TypeError) << "Option '" << key << "': expected type '"
                                 << ffi::TypeTraits<T>::TypeStr() << "' but got '"
                                 << val.GetTypeKey() << "'";
      }
      return ffi::Any(opt.value());
    });
  }

  template <typename Trait>
  static void ApplyTrait(OptionEntry* entry, ffi::reflection::FieldInfoBuilder* info,
                         Trait&& trait) {
    using T = std::decay_t<Trait>;
    if constexpr (std::is_same_v<T, AttrValidator>) {
      entry->validator = std::move(trait.func);
    } else if constexpr (std::is_base_of_v<ffi::reflection::InfoTrait, T>) {
      trait.Apply(info);
    } else if constexpr (std::is_same_v<T, const char*> || std::is_same_v<T, char*>) {
      const char* doc = trait;
      if (doc != nullptr && doc[0] != '\0') {
        info->doc = TVMFFIByteArray{doc, std::char_traits<char>::length(doc)};
      }
    }
  }

  template <typename T, typename... Traits>
  OptionEntry MakeEntry(const ffi::String& key, Traits&&... traits) {
    OptionEntry e;
    e.key = key;
    e.type_str = ffi::String(ffi::TypeTraits<T>::TypeStr());
    e.validator = MakeValidator<T>(key);
    // Apply traits through a temporary FieldInfoBuilder so existing
    // reflection traits (notably refl::DefaultValue) are reused unchanged.
    ffi::reflection::FieldInfoBuilder info{};
    info.flags = 0;
    info.default_value = ffi::AnyView(nullptr).CopyToTVMFFIAny();
    info.doc = TVMFFIByteArray{nullptr, 0};
    (ApplyTrait(&e, &info, std::forward<Traits>(traits)), ...);
    if (info.flags & kTVMFFIFieldFlagBitMaskHasDefault) {
      e.has_default = true;
      e.default_value = ffi::AnyView::CopyFromTVMFFIAny(info.default_value);
      // Release the extra ref created by CopyToTVMFFIAny in Apply
      if (info.default_value.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin) {
        ffi::details::ObjectUnsafe::DecRefObjectHandle(info.default_value.v_obj);
      }
    }
    return e;
  }

  /*! \brief Declared options in declaration order. */
  std::vector<OptionEntry> options_;
  /*! \brief Map from key string to index in options_. */
  std::unordered_map<std::string, size_t> key_to_index_;
  /*! \brief Optional whole-config canonicalizer. */
  Canonicalizer canonicalizer_{nullptr};
  /*! \brief Whether unknown keys trigger an error. */
  bool error_on_unknown_ = true;
};

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_CONFIG_SCHEMA_H_
