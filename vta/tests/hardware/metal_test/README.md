
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Running Instruction Level Metal Tests 
======================
Running metal tests on instruction level is crucial for hardware, and constructing those tests are tedious in C. Therefore,
having a user friendly interface instruction level test is convenient for verification. This is a python wrapper provides 
that functionallity to VTA.

**Run Test**
* run 'sudo make' to create shared library and macros
* two examples are provided: `alu.py` and `gemm.py`

**Some Notes for adding additional functions**
* to add C++ functions, remember to add the `extern "C"` header to prevent the compiler from modifying the function name
* make sure to wrap the function in `insn_lib.py` to have a simple type check

