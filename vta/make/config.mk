#-------------------------------------------------------------------------------
#  Template configuration for compiling VTA runtime.
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of nnvm. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#-------------------------------------------------------------------------------

#---------------------
# choice of compiler
#--------------------

# the additional link flags you want to add
ADD_LDFLAGS=

# the additional compile flags you want to add
ADD_CFLAGS=
