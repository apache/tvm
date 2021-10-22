
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#
# Common options
#

ifneq ($V,1)
q := @
else
q := 
endif


define set_cm
sm := $1
$1_cflags := $2
$1_remove_cflags := $3
endef

# Macro to set the rules to build a specific C file
#-------------------------------------------------------------------------------
# Arguments are:
# 1 - C source file
# 2 - Destination file without extension (generated object)
# 
# Global definitions:
# 	sm 		- group name, ${sm}_remove_cflags, ${sm}_cflags
# 	CFLAGS	- Pre and compiler options
define process_c_srcs

#(info Rule for $(1) $(2).o)
objs += $(2).o

$(eval _base_ := $(notdir $(2)))
$(eval comp-dep-$(_base_) := $(2).d)
$(eval comp-i-$(_base_) := $(2).i)

ifeq (${${sm}_remove_cflags},)
${sm}_remove_cflags :=
endif

comp-cflags-$(_base_) := $(CFLAGS) $${cflags-${_base_}.c}
comp-cflags-$(_base_) := $$(filter-out $${${sm}_remove_cflags}, $${comp-cflags-$(_base_)})
comp-cflags-$(_base_) += $(${sm}_cflags)

comp-iflags-$(_base_) := $$(comp-cflags-$(_base_))
comp-cflags-$(_base_) += -MD -MF $$(comp-dep-$(_base_)) -MT $(2).o

clean-files += $(2).o # $$(comp-dep-$(_base_)) $(2).su $$(comp-i-$(_base_))
clean-files-dep += $$(comp-dep-$(_base_))
clean-files-su += $(2).su
clean-files-i += $$(comp-i-$(_base_))

-include $$(comp-dep-$(_base_))

$(2).o: $(1)
	@echo "CC(${sm})  $$<"
	@mkdir -p $$(@D)
	$(q)$(CC) $$(comp-iflags-$(_base_)) -E $$< -o $$(comp-i-$(_base_))
	$(q)$(CC) $$(comp-cflags-$(_base_)) -c $$< -o $$@
endef

# Macro to set the rules to build a specific S file
#-------------------------------------------------------------------------------
# Arguments are:
# 1 - S source file
# 2 - Destination file without extension (generated object)
# 
# Global definitions:
# 	sm 		- group name
# 	ASFLAGS	- Pre and compiler options
define process_s_srcs

#(info Rule for $(1) $(2))
objs += $(2).o

$(eval _base_ := $(notdir $(2)))
$(eval comp-dep-$(_base_) := $(2).d)
$(eval comp-i-$(_base_) := $(2).i)

comp-asflags-$(_base_) := $(ASFLAGS)

clean-files += $(2).o

-include $$(comp-dep-$(_base_))

$(2).o: $(1)
	@echo "AS(${sm})  $$<"
	@mkdir -p $$(@D)
	$(q)$(CC) $$(comp-asflags-$(_base_)) -c $$< -o $$@
endef


# Macro to set the rules to build a specific CC file
#-------------------------------------------------------------------------------
# Arguments are:
# 1 - C++ source file
# 2 - Destination file without extension (generated object)
# 
# Global definitions:
# 	sm 		 - group name, ${sm}_remove_cxxflags, ${sm}_cxxflags
# 	CXXFLAG  - Pre and compiler options
define process_cc_srcs

#(info Rule for $(1) $(2))
objs += $(2).o

$(eval _base_ := $(notdir $(2)))
$(eval comp-dep-$(_base_) := $(2).d)
$(eval comp-i-$(_base_) := $(2).i)

ifeq (${${sm}_remove_cxxflags},)
${sm}_remove_cxxflags :=
endif

comp-cxxflags-$(_base_) := $(CXXFLAGS) $${cxxflags-${_base_}.cc}
comp-cxxflags-$(_base_) := $$(filter-out $${${sm}_remove_cxxflags}, $${comp-cxxflags-$(_base_)})
comp-cxxflags-$(_base_) += $(${sm}_cxxflags)

comp-ixxflags-$(_base_) := $$(comp-cxxflags-$(_base_))
comp-cxxflags-$(_base_) += -MD -MF $$(comp-dep-$(_base_)) -MT $(2).o

clean-files += $(2).o # $$(comp-dep-$(_base_)) $(2).su $$(comp-i-$(_base_))
clean-files-dep += $$(comp-dep-$(_base_))
clean-files-su += $(2).su
clean-files-i += $$(comp-i-$(_base_))

-include $$(comp-dep-$(_base_))

$(2).o: $(1)
	@echo "CXX(${sm}) $$<"
	@mkdir -p $$(@D)
	$(q)$(CXX) $$(comp-ixxflags-$(_base_)) -E $$< -o $$(comp-i-$(_base_))
	$(q)$(CXX) $$(comp-cxxflags-$(_base_)) -c $$< -o $$@
endef
